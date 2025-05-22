import os
import subprocess
import signal
import atexit
import time
from pathlib import Path
from app.logger import logger

class SSHTunnel:
    def __init__(self):
        self.ssh_host = os.environ.get('SSH_HOST')
        self.ssh_private_key = os.environ.get('SSH_PRIVATE_KEY')
        self.key_path = '/tmp/runpod_key'
        self.process = None
        
    def start(self):
        """Start SSH tunnel to RunPod"""
        if not self.ssh_host or not self.ssh_private_key:
            logger.warning("SSH_HOST or SSH_PRIVATE_KEY not set, skipping tunnel setup")
            return False
            
        try:
            # Write private key to file
            with open(self.key_path, 'w') as f:
                f.write(self.ssh_private_key)
            os.chmod(self.key_path, 0o600)
            
            # Start SSH tunnel
            self.process = subprocess.Popen([
                'ssh',
                '-N',
                '-o', 'StrictHostKeyChecking=no',
                '-L', '8888:localhost:8888',
                self.ssh_host,
                '-i', self.key_path
            ], stdout=subprocess.PIPE, stderr=subprocess.PIPE)
            
            # Wait a moment for tunnel to establish
            time.sleep(2)
            
            if self.process.poll() is not None:
                stderr = self.process.stderr.read().decode('utf-8')
                logger.error(f"SSH tunnel failed to start: {stderr}")
                return False
                
            logger.info(f"SSH tunnel established to {self.ssh_host}")
            
            # Register cleanup on exit
            atexit.register(self.stop)
            return True
            
        except Exception as e:
            logger.error(f"Error setting up SSH tunnel: {str(e)}")
            self.cleanup()
            return False
            
    def stop(self):
        """Stop SSH tunnel"""
        if self.process:
            self.process.terminate()
            self.process = None
        self.cleanup()
        
    def cleanup(self):
        """Clean up key file"""
        try:
            if Path(self.key_path).exists():
                os.unlink(self.key_path)
        except Exception as e:
            logger.error(f"Error cleaning up SSH key: {str(e)}")

# Create singleton instance
tunnel = SSHTunnel()
