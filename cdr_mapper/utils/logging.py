"""
Logging utilities for CDR Mapper application.

Provides centralized logging for user actions and errors during map rendering.
Supports both local file logging and S3 logging for AWS deployments.
"""

import os
import traceback
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Optional

try:
    import boto3

    S3_AVAILABLE = True
except ImportError:
    S3_AVAILABLE = False


class MapLogger:
    """Logger for tracking map loading, layer selections, and errors."""

    def __init__(
        self, log_dir: Path, s3_bucket: Optional[str] = None, s3_prefix: str = "logs"
    ):
        """
        Initialize the map logger.

        Parameters:
        -----------
        log_dir : Path
            Directory where log files will be stored locally
        s3_bucket : str, optional
            S3 bucket name for remote logging. If provided, logs will be written to S3.
        s3_prefix : str
            S3 key prefix for log files (default: "logs")
        """
        self.log_dir = Path(log_dir)
        self.log_dir.mkdir(parents=True, exist_ok=True)
        self.log_file = self.log_dir / "cdr_mapper.log"
        self.session_start = datetime.now()

        # S3 configuration
        self.s3_bucket = s3_bucket
        self.s3_prefix = s3_prefix
        self.s3_client = None
        self.s3_key = None

        if s3_bucket and S3_AVAILABLE:
            try:
                self.s3_client = boto3.client("s3")
                # Test S3 access
                self.s3_client.head_bucket(Bucket=s3_bucket)
                self.s3_key = f"{s3_prefix}/cdr_mapper.log"

                # Download existing S3 log to local file if it exists (preserve history)
                try:
                    response = self.s3_client.get_object(
                        Bucket=s3_bucket, Key=self.s3_key
                    )
                    existing_content = response["Body"].read().decode("utf-8")
                    # Write existing S3 content to local file
                    with open(self.log_file, "w", encoding="utf-8") as f:
                        f.write(existing_content)
                    print(
                        f"Loaded existing log from S3 ({len(existing_content)} bytes)"
                    )
                except self.s3_client.exceptions.NoSuchKey:
                    print("No existing S3 log found, starting fresh")
                except Exception as e:
                    print(f"Warning: Could not download existing S3 log: {e}")

            except Exception as e:
                print(f"Warning: Could not connect to S3 bucket {s3_bucket}: {e}")
                print("Falling back to local logging only")
                self.s3_client = None

    def _timestamp(self) -> str:
        """Get current timestamp in ISO format."""
        return datetime.now().isoformat(sep=" ", timespec="seconds")

    def _write_line(self, message: str):
        """Write a line to both local file and S3 (if configured)."""
        # Write to local file (append mode)
        with open(self.log_file, "a", encoding="utf-8") as f:
            f.write(f"{message}\n")

        # Write to S3 if configured
        if self.s3_client and self.s3_bucket and self.s3_key:
            try:
                # Read current log file content (includes history from S3 + new entries)
                with open(self.log_file, "r", encoding="utf-8") as f:
                    log_content = f.read()

                # Upload entire log to S3 (overwrites, but includes full history)
                self.s3_client.put_object(
                    Bucket=self.s3_bucket,
                    Key=self.s3_key,
                    Body=log_content,
                    ContentType="text/plain",
                )
            except Exception as e:
                # Don't fail the application if S3 logging fails
                print(f"Warning: Failed to write log to S3: {e}")

    def log_session_start(self):
        """Log the start of a new map loading session."""
        separator = "=" * 80
        self._write_line(f"\n{separator}")
        self._write_line(f"[{self._timestamp()}] NEW SESSION STARTED")
        self._write_line(separator)

    def log_layer_selection(self, selected_layers: Dict[str, bool]):
        """
        Log which layers are selected/unselected.

        Parameters:
        -----------
        selected_layers : dict
            Dictionary of layer_key -> bool (selected or not)
        """
        self._write_line(f"\n[{self._timestamp()}] LAYER SELECTION:")

        # Separate into selected and unselected
        selected = [k for k, v in selected_layers.items() if v]
        unselected = [k for k, v in selected_layers.items() if not v]

        if selected:
            self._write_line(f"  Selected layers ({len(selected)}):")
            for layer in sorted(selected):
                self._write_line(f"    ✓ {layer}")
        else:
            self._write_line("  No layers selected")

        if unselected:
            self._write_line(f"  Unselected layers ({len(unselected)}):")
            for layer in sorted(unselected):
                self._write_line(f"    ✗ {layer}")

    def log_layer_load_start(self, layer_key: str, layer_name: str):
        """
        Log the start of loading a layer.

        Parameters:
        -----------
        layer_key : str
            Layer identifier (e.g., "storage.basaltic")
        layer_name : str
            Human-readable layer name
        """
        self._write_line(
            f"\n[{self._timestamp()}] Loading layer: {layer_name} ({layer_key})"
        )

    def log_layer_load_success(self, layer_key: str, data_info: str):
        """
        Log successful layer loading.

        Parameters:
        -----------
        layer_key : str
            Layer identifier
        data_info : str
            Information about the loaded data (e.g., "raster 1000x2000", "5000 features")
        """
        self._write_line(
            f"[{self._timestamp()}] ✓ Successfully loaded {layer_key}: {data_info}"
        )

    def log_layer_load_error(
        self,
        layer_key: str,
        layer_name: str,
        error: Exception,
        include_traceback: bool = True,
    ):
        """
        Log an error during layer loading.

        Parameters:
        -----------
        layer_key : str
            Layer identifier
        layer_name : str
            Human-readable layer name
        error : Exception
            The exception that occurred
        include_traceback : bool
            Whether to include full traceback (default: True)
        """
        self._write_line(
            f"\n[{self._timestamp()}] ✗ ERROR loading {layer_name} ({layer_key})"
        )
        self._write_line(f"  Error type: {type(error).__name__}")
        self._write_line(f"  Error message: {str(error)}")

        if include_traceback:
            self._write_line("  Traceback:")
            tb_lines = traceback.format_exception(
                type(error), error, error.__traceback__
            )
            for line in tb_lines:
                for subline in line.rstrip().split("\n"):
                    self._write_line(f"    {subline}")

    def log_map_render_start(self, num_layers: int):
        """
        Log the start of map rendering.

        Parameters:
        -----------
        num_layers : int
            Number of layers to render
        """
        self._write_line(
            f"\n[{self._timestamp()}] RENDERING MAP with {num_layers} layer(s)"
        )

    def log_map_render_success(self):
        """Log successful map rendering."""
        self._write_line(f"[{self._timestamp()}] ✓ Map rendered successfully")

    def log_map_render_error(self, error: Exception, include_traceback: bool = True):
        """
        Log an error during map rendering.

        Parameters:
        -----------
        error : Exception
            The exception that occurred
        include_traceback : bool
            Whether to include full traceback (default: True)
        """
        self._write_line(f"\n[{self._timestamp()}] ✗ ERROR rendering map")
        self._write_line(f"  Error type: {type(error).__name__}")
        self._write_line(f"  Error message: {str(error)}")

        if include_traceback:
            self._write_line("  Traceback:")
            tb_lines = traceback.format_exception(
                type(error), error, error.__traceback__
            )
            for line in tb_lines:
                for subline in line.rstrip().split("\n"):
                    self._write_line(f"    {subline}")

    def log_custom_message(self, message: str, level: str = "INFO"):
        """
        Log a custom message.

        Parameters:
        -----------
        message : str
            Message to log
        level : str
            Log level (INFO, WARNING, ERROR, etc.)
        """
        self._write_line(f"[{self._timestamp()}] [{level}] {message}")

    def log_session_end(self):
        """Log the end of a session."""
        duration = datetime.now() - self.session_start
        self._write_line(
            f"\n[{self._timestamp()}] SESSION ENDED (duration: {duration})"
        )
        separator = "=" * 80
        self._write_line(separator)


def get_logger(
    log_dir: Optional[Path] = None,
    s3_bucket: Optional[str] = None,
    s3_prefix: str = "logs",
) -> MapLogger:
    """
    Get or create a logger instance.

    Parameters:
    -----------
    log_dir : Path, optional
        Directory for log files. If None, uses ./logs relative to cwd
    s3_bucket : str, optional
        S3 bucket name for remote logging. If None, checks CDR_MAPPER_LOG_BUCKET env var.
    s3_prefix : str
        S3 key prefix for log files (default: "logs")

    Returns:
    --------
    MapLogger instance
    """
    if log_dir is None:
        log_dir = Path.cwd() / "logs"

    # Check environment variable for S3 bucket if not provided
    if s3_bucket is None:
        s3_bucket = os.getenv("CDR_MAPPER_LOG_BUCKET")

    return MapLogger(log_dir, s3_bucket=s3_bucket, s3_prefix=s3_prefix)
