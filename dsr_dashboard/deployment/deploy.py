import argparse
import subprocess
import os
import json

def deploy(environment):
    """Deploy to specified environment"""
    if environment not in ['test', 'prod']:
        raise ValueError("Environment must be 'test' or 'prod'")
    
    # 2. Run existing deployment script with environment
    subprocess.run([
        './deployment/deploy_lambdas.sh',
        environment
    ])

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('environment', choices=['test', 'prod'])
    args = parser.parse_args()
    
    deploy(args.environment)

if __name__ == '__main__':
    main()
