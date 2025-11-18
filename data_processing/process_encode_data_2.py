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
    ('ENCSR000DWB', ['ENCFF816ECC', 'ENCFF035SOZ']),
    ('ENCSR000APC', ['ENCFF874SMO', 'ENCFF007YZT']),
    ('ENCSR000EWC', ['ENCFF229LAF', 'ENCFF089KHK']),
    ('ENCSR000AKT', ['ENCFF446FUS', 'ENCFF583GSH']),
    ('ENCSR000APD', ['ENCFF711PLM', 'ENCFF649PBO', 'ENCFF132YSI']),
    ('ENCSR000AKX', ['ENCFF255QRL', 'ENCFF923YUN']),
    ('ENCSR000EVZ', ['ENCFF669AOZ', 'ENCFF946AXU']),
    ('ENCSR000APE', ['ENCFF104THG', 'ENCFF155UQU']),
    ('ENCSR057BWO', ['ENCFF911YNM', 'ENCFF280HZC']),
    ('ENCSR000DRX', ['ENCFF265UBT', 'ENCFF824VSE']),
    ('ENCSR000AKE', ['ENCFF353YPB', 'ENCFF677MAG']),
    ('ENCSR000AOV', ['ENCFF703AIM', 'ENCFF256UUQ']),
    ('ENCSR000AKF', ['ENCFF047NLO', 'ENCFF385FLM']),
    ('ENCSR000AKG', ['ENCFF008LEY', 'ENCFF081ODV']),
    ('ENCSR000AOW', ['ENCFF465NXQ', 'ENCFF967CHL']),
    ('ENCSR000AKI', ['ENCFF132EDT', 'ENCFF334NVA']),
    ('ENCSR000AKH', ['ENCFF729HQN', 'ENCFF405REQ']),
    ('ENCSR000AOX', ['ENCFF306ENK', 'ENCFF889UPU', 'ENCFF349VRN']),
    ('ENCSR000DUF', ['ENCFF350FDO', 'ENCFF570RYZ']),
    ('ENCSR000AOL', ['ENCFF027OKJ', 'ENCFF093BIB', 'ENCFF369DOB']),
    ('ENCSR000AMB', ['ENCFF129YEO', 'ENCFF211APO', 'ENCFF080RGC']),
    ('ENCSR000AOK', ['ENCFF830TZW']),
    ('ENCSR000APV', ['ENCFF176CMB', 'ENCFF678ZFC']),
    ('ENCSR000AMC', ['ENCFF210WWS', 'ENCFF884OTY']),
    ('ENCSR000AOM', ['ENCFF074WLE']),
    ('ENCSR000AMQ', ['ENCFF794BQI', 'ENCFF042DLI']),
    ('ENCSR000AMD', ['ENCFF312VEN', 'ENCFF855SBG']),
    ('ENCSR000ATD', ['ENCFF828OFD', 'ENCFF044CBD']),
    ('ENCSR882DWM', ['ENCFF102ASX', 'ENCFF892MBZ']),
    ('ENCSR014ARU', ['ENCFF955ZRZ', 'ENCFF863TKH']),
    ('ENCSR000BSO', ['ENCFF621SYN', 'ENCFF226CAN']),
    ('ENCSR000BQK', ['ENCFF856BLZ', 'ENCFF463JZL']),
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
