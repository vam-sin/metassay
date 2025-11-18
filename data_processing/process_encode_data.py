import subprocess
import os
from multiprocessing import Pool
import sys

def process_experiment(exp_data):
    """Process a single experiment: download BAMs, run bam2bw, cleanup"""
    exp_id, bam_ids, base_dir = exp_data
    
    try:
        print(f"Starting {exp_id}...")
        
        # Create directory with absolute path
        exp_dir = os.path.join(base_dir, exp_id)
        os.makedirs(exp_dir, exist_ok=True)
        
        # Remove old bigwig files
        subprocess.run(f'rm -f {exp_dir}/aln_*.bw', shell=True, check=False)
        
        # Download BAM files
        bam_files = []
        for bam_id in bam_ids:
            bam_file = os.path.join(exp_dir, f"aln_{bam_id}.bam")
            url = f"https://www.encodeproject.org/files/{bam_id}/@@download/{bam_id}.bam"
            print(f"  {exp_id}: Downloading {bam_id}...")
            result = subprocess.run(['wget', url, '-O', bam_file], 
                                  capture_output=True, text=True)
            if result.returncode != 0:
                print(f"  {exp_id}: Failed to download {bam_id}")
                return f"{exp_id}: FAILED (download)"
            bam_files.append(bam_file)
        
        # Run bam2bw
        print(f"  {exp_id}: Running bam2bw...")
        chrom_sizes = os.path.join(base_dir, 'hg38.chrom.sizes')
        output_name = os.path.join(exp_dir, f'aln_{exp_id}')
        cmd = ['bam2bw'] + bam_files + ['-s', chrom_sizes, '-n', output_name, '-v', '-u']
        result = subprocess.run(cmd, capture_output=True, text=True)
        if result.returncode != 0:
            print(f"  {exp_id}: bam2bw failed: {result.stderr}")
            return f"{exp_id}: FAILED (bam2bw)"
        
        # Remove BAM files
        for bam_file in bam_files:
            os.remove(bam_file)
        
        print(f"  {exp_id}: COMPLETED")
        return f"{exp_id}: SUCCESS"
        
    except Exception as e:
        print(f"  {exp_id}: ERROR - {str(e)}")
        return f"{exp_id}: FAILED ({str(e)})"

# All experiments data: (exp_id, [bam_ids])
experiments = [
    ('ENCSR000EKQ', ['ENCFF708UMH']),
    ('ENCSR000EMT', ['ENCFF020WZB', 'ENCFF729UYK']),
    ('ENCSR149XIL', ['ENCFF474LSZ', 'ENCFF839SPF']),
    ('ENCSR868FGK', ['ENCFF534DCE', 'ENCFF128WZG', 'ENCFF077FBI']),
    ('ENCSR637XSC', ['ENCFF981FXV', 'ENCFF440GRZ', 'ENCFF962FMH']),
    ('ENCSR291GJU', ['ENCFF990VCP', 'ENCFF624SON', 'ENCFF926KFU']),
    ('ENCSR388QZF', ['ENCFF480AJZ', 'ENCFF785OCU']),
    ('ENCSR000EGM', ['ENCFF172KOJ', 'ENCFF265ZSP']),
    ('ENCSR000EGE', ['ENCFF857UQA', 'ENCFF748JCP']),
    ('ENCSR075HTM', ['ENCFF570EKM', 'ENCFF613CGQ']),
    ('ENCSR946BXO', ['ENCFF062SAB', 'ENCFF944OZW']),
    ('ENCSR000EGN', ['ENCFF570DCQ', 'ENCFF208UMS']),
    ('ENCSR948VFL', ['ENCFF568KYJ', 'ENCFF515ASV']),
    ('ENCSR158LJN', ['ENCFF428YJS', 'ENCFF870SUX']),
    ('ENCSR000BGD', ['ENCFF501USI', 'ENCFF886CYK']),
    ('ENCSR000DZN', ['ENCFF747TJH', 'ENCFF876QCV']),
    ('ENCSR000BUF', ['ENCFF538DUT', 'ENCFF108DXA']),
    ('ENCSR000DZD', ['ENCFF363VYY', 'ENCFF791IGI']),
    ('ENCSR330OEO', ['ENCFF434QLO', 'ENCFF121LBW']),
    ('ENCSR000DZS', ['ENCFF912ING', 'ENCFF455IBT']),
    ('ENCSR000DYS', ['ENCFF078GJW', 'ENCFF891HXN']),
    ('ENCSR441VHN', ['ENCFF487CFV', 'ENCFF247FMB']),
    ('ENCSR000DZF', ['ENCFF456EJZ', 'ENCFF883PZJ']),
    ('ENCSR000EEM', ['ENCFF835GBL', 'ENCFF845YGC']),
    ('ENCSR607XFI', ['ENCFF669HQI', 'ENCFF015YAX']),
    ('ENCSR908HWZ', ['ENCFF680QZI', 'ENCFF086LBT']),
    ('ENCSR112ALD', ['ENCFF629RNF', 'ENCFF906EDB']),
    ('ENCSR271XMW', ['ENCFF779LBX', 'ENCFF320EIJ']),
    ('ENCSR337NWW', ['ENCFF415PXR', 'ENCFF465PUO']),
    ('ENCSR343RJR', ['ENCFF555KBC', 'ENCFF299TVJ']),
    ('ENCSR282ZLP', ['ENCFF363YHY', 'ENCFF574DLO']),
    ('ENCSR278JQG', ['ENCFF694WGO', 'ENCFF296HBB']),
    ('ENCSR000BTM', ['ENCFF279PNE', 'ENCFF749EPE']),
    ('ENCSR000DWD', ['ENCFF440ARP', 'ENCFF656DMV']),
    ('ENCSR000EWB', ['ENCFF652TXG', 'ENCFF508LLH']),
]

if __name__ == '__main__':
    # Test bam2bw is available
    result = subprocess.run(['bam2bw', '--help'], capture_output=True, text=True)
    if result.returncode != 0:
        print("✗ bam2bw not found or not working")
        sys.exit(1)
    print("✓ bam2bw is available and working!")
    
    # Setup the dataset directory
    base_dir = '/rcp/nallapar/cshl/data_processing/encode_dataset/dataset'
    os.makedirs(base_dir, exist_ok=True)
    
    # Check if hg38.chrom.sizes exists
    chrom_sizes = os.path.join(base_dir, 'hg38.chrom.sizes')
    if not os.path.exists(chrom_sizes):
        print(f"Warning: {chrom_sizes} not found. Make sure it exists before running.")
    
    print(f"Working directory: {base_dir}")
    print(f"Processing {len(experiments)} experiments in parallel...")
    
    # Prepare experiment data with base_dir
    exp_data_with_dir = [(exp_id, bam_ids, base_dir) for exp_id, bam_ids in experiments]
    
    # Use multiprocessing pool to process experiments in parallel
    # Adjust the number of processes based on your system
    num_processes = min(16, len(experiments))  # Use up to 16 processes
    
    with Pool(processes=num_processes) as pool:
        results = pool.map(process_experiment, exp_data_with_dir)
    
    # Print summary
    print("\n" + "="*60)
    print("SUMMARY")
    print("="*60)
    successes = sum(1 for r in results if 'SUCCESS' in r)
    failures = sum(1 for r in results if 'FAILED' in r)
    
    print(f"Total: {len(results)}")
    print(f"Successful: {successes}")
    print(f"Failed: {failures}")
    
    if failures > 0:
        print("\nFailed experiments:")
        for r in results:
            if 'FAILED' in r:
                print(f"  {r}")
    
    print("="*60)
