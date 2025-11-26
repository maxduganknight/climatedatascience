#!/usr/bin/env python3
import argparse
import subprocess
import os
import sys

def deploy(environment):
    """Deploy to specified environment"""
    if environment not in ['test', 'prod']:
        raise ValueError("Environment must be 'test' or 'prod'")
    
    # Get the directory where this script is located
    script_dir = os.path.dirname(os.path.abspath(__file__))
    
    # Get the project root (parent of deployment dir)
    project_dir = os.path.dirname(script_dir)
    
    deploy_script = os.path.join(script_dir, 'deploy_cdr_lambda.sh')
    
    # Check if deploy script exists
    if not os.path.exists(deploy_script):
        print(f"ERROR: Deployment script not found at {deploy_script}")
        print("Please create the script with the provided content.")
        sys.exit(1)
    
    # Make the script executable
    subprocess.run(['chmod', '+x', deploy_script], check=True)
    
    # Run the deployment script
    process = subprocess.run(
        [deploy_script, environment],
        cwd=project_dir,
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
        universal_newlines=True,
        bufsize=1
    )
    
    # Print output and check status
    output = process.stdout
    print(output)
    
    if process.returncode != 0:
        print(f"Deployment failed with return code: {process.returncode}")
        sys.exit(1)
    
    print("Deployment completed successfully!")

def main():
    parser = argparse.ArgumentParser(description='Deploy CDR Lambda function')
    parser.add_argument('environment', choices=['test', 'prod'], 
                        help='Environment to deploy to (test or prod)')
    args = parser.parse_args()
    
    deploy(args.environment)

if __name__ == '__main__':
    main()