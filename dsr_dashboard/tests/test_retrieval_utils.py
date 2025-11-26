import pytest
import json
import os
from unittest.mock import patch, mock_open, MagicMock
from pathlib import Path
from datetime import datetime, timedelta, date
import pandas as pd
from io import StringIO
from utils.retrieval_utils import (
    get_aws_secret, load_config, needs_update, get_files_list,
    get_file_date, save_dataset, decimal_year_to_datetime, cleanup_old_files
)

class TestGetAwsSecret:
    @patch.dict(os.environ, {'ENVIRONMENT': 'test'})
    @patch('boto3.session.Session')
    def test_get_aws_secret_success(self, mock_session):
        """Test successful secret retrieval"""
        mock_client = MagicMock()
        mock_session.return_value.client.return_value = mock_client
        mock_client.get_secret_value.return_value = {
            'SecretString': '{"api_key": "test123", "user": "test_user"}'
        }
        
        result = get_aws_secret()
        
        assert result['api_key'] == 'test123'
        assert result['user'] == 'test_user'
        mock_client.get_secret_value.assert_called_once_with(
            SecretId='dashboard-creds-test'
        )

    @patch.dict(os.environ, {'ENVIRONMENT': 'prod'})
    @patch('boto3.session.Session')
    def test_get_aws_secret_error(self, mock_session):
        """Test secret retrieval error handling"""
        mock_client = MagicMock()
        mock_session.return_value.client.return_value = mock_client
        mock_client.get_secret_value.side_effect = Exception("Access denied")
        
        with pytest.raises(Exception):
            get_aws_secret()

class TestLoadConfig:
    @patch('builtins.open', new_callable=mock_open, read_data='{"test_dataset": {"update_cadence": "daily"}}')
    def test_load_config_success(self, mock_file):
        """Test successful config loading"""
        result = load_config()
        
        assert 'test_dataset' in result
        assert result['test_dataset']['update_cadence'] == 'daily'

    @patch('builtins.open', side_effect=FileNotFoundError())
    def test_load_config_file_not_found(self, mock_file):
        """Test config loading when file doesn't exist"""
        with pytest.raises(FileNotFoundError):
            load_config()

    @patch('builtins.open', new_callable=mock_open, read_data='{"invalid": json}')
    def test_load_config_invalid_json(self, mock_file):
        """Test handling of malformed JSON in config"""
        with pytest.raises(json.JSONDecodeError):
            load_config()

class TestGetFilesList:
    @patch('glob.glob')
    def test_get_files_list_local(self, mock_glob):
        """Test getting local file list"""
        mock_glob.return_value = ['/path/to/test_dataset_20240101.csv', '/path/to/test_dataset_20240102.csv']
        
        result = get_files_list('test_dataset', is_local=True)
        
        assert len(result) == 2
        assert '/path/to/test_dataset_20240101.csv' in result

    @patch('boto3.client')
    def test_get_files_list_s3(self, mock_boto):
        """Test getting S3 file list"""
        mock_s3 = MagicMock()
        mock_boto.return_value = mock_s3
        mock_s3.list_objects_v2.return_value = {
            'Contents': [
                {'Key': 'processed/test_dataset_20240101.csv'},
                {'Key': 'processed/test_dataset_20240102.csv'}
            ]
        }
        
        result = get_files_list('test_dataset', is_local=False)
        
        assert len(result) == 2
        assert 'processed/test_dataset_20240101.csv' in result

    @patch('boto3.client')
    def test_get_files_list_s3_empty(self, mock_boto):
        """Test getting empty S3 file list"""
        mock_s3 = MagicMock()
        mock_boto.return_value = mock_s3
        mock_s3.list_objects_v2.return_value = {}
        
        result = get_files_list('test_dataset', is_local=False)
        
        assert len(result) == 0

class TestGetFileDate:
    @patch('utils.retrieval_utils.get_files_list')
    def test_get_file_date_single_file(self, mock_get_files):
        """Test extracting date from single file"""
        mock_get_files.return_value = ['/path/to/test_dataset_20240315.csv']
        
        result = get_file_date('test_dataset')
        
        assert result == date(2024, 3, 15)

    @patch('utils.retrieval_utils.get_files_list')
    def test_get_file_date_no_files(self, mock_get_files):
        """Test behavior when no files exist"""
        mock_get_files.return_value = []
        
        result = get_file_date('test_dataset')
        
        assert result is None

    @patch('utils.retrieval_utils.get_files_list')
    def test_get_file_date_multiple_files_error(self, mock_get_files):
        """Test error when multiple files exist"""
        mock_get_files.return_value = [
            '/path/to/test_dataset_20240315.csv',
            '/path/to/test_dataset_20240316.csv'
        ]
        
        with pytest.raises(RuntimeError, match="Multiple files found"):
            get_file_date('test_dataset')

    @patch('utils.retrieval_utils.get_files_list')
    def test_get_file_date_invalid_format(self, mock_get_files):
        """Test error with invalid date format"""
        mock_get_files.return_value = ['/path/to/test_dataset_invalid.csv']
        
        with pytest.raises(ValueError, match="Invalid date format"):
            get_file_date('test_dataset')

class TestNeedsUpdate:
    @patch('utils.retrieval_utils.get_file_date')
    def test_needs_update_no_files(self, mock_get_date):
        """Test update needed when no files exist"""
        mock_get_date.return_value = None
        config = {'test_dataset': {'update_cadence': 'daily'}}
        
        result = needs_update('test_dataset', config, True)
        
        assert result is True

    @patch('utils.retrieval_utils.get_file_date')
    def test_needs_update_daily_old_file(self, mock_get_date):
        """Test daily update logic with old file"""
        old_date = date.today() - timedelta(days=2)
        mock_get_date.return_value = old_date
        config = {'test_dataset': {'update_cadence': 'daily'}}
        
        result = needs_update('test_dataset', config, True)
        
        assert result is True

    @patch('utils.retrieval_utils.get_file_date')
    def test_needs_update_daily_recent_file(self, mock_get_date):
        """Test daily update logic with recent file"""
        recent_date = date.today()
        mock_get_date.return_value = recent_date
        config = {'test_dataset': {'update_cadence': 'daily'}}
        
        result = needs_update('test_dataset', config, True)
        
        assert result is False

    @pytest.mark.parametrize("cadence,days_old,should_update", [
        ('daily', 0, False),
        ('daily', 1, True),
        ('weekly', 6, False),
        ('weekly', 7, True),
        ('monthly', 29, False),
        ('monthly', 30, True),
        ('quarterly', 89, False),
        ('quarterly', 90, True),
        ('annually', 364, False),
        ('annually', 365, True),
    ])
    @patch('utils.retrieval_utils.get_file_date')
    def test_needs_update_various_cadences(self, mock_get_date, cadence, days_old, should_update):
        """Parameterized test for different update cadences"""
        old_date = date.today() - timedelta(days=days_old)
        mock_get_date.return_value = old_date
        config = {'test_dataset': {'update_cadence': cadence}}
        
        result = needs_update('test_dataset', config, True)
        
        assert result == should_update

    @patch('utils.retrieval_utils.get_file_date')
    def test_needs_update_missing_dataset_error(self, mock_get_date):
        """Test that function returns True when dataset not in config (due to exception handling)"""
        # Mock files exist but dataset not in config
        mock_get_date.return_value = date.today() - timedelta(days=5)
        config = {'other_dataset': {'update_cadence': 'daily'}}
        
        # Should return True (defaults to update on error)
        result = needs_update('missing_dataset', config, True)
        assert result is True

    @patch('utils.retrieval_utils.get_file_date')
    def test_needs_update_unknown_cadence_defaults_monthly(self, mock_get_date):
        """Test unknown cadence defaults to monthly (30 days)"""
        old_date = date.today() - timedelta(days=31)
        mock_get_date.return_value = old_date
        config = {'test_dataset': {'update_cadence': 'unknown_cadence'}}
        
        result = needs_update('test_dataset', config, True)
        
        assert result is True

class TestDecimalYearToDatetime:
    def test_decimal_year_to_datetime_basic(self):
        """Test basic decimal year conversion"""
        result = decimal_year_to_datetime(2020.5)
        
        # Should be roughly mid-year 2020
        assert result.year == 2020
        assert result.month >= 6  # Around July

    def test_decimal_year_to_datetime_start_of_year(self):
        """Test conversion for start of year"""
        result = decimal_year_to_datetime(2020.0)
        
        assert result.year == 2020
        assert result.month == 1
        assert result.day == 1

    def test_decimal_year_to_datetime_leap_year(self):
        """Test conversion for leap year"""
        result = decimal_year_to_datetime(2020.25)  # Quarter into leap year
        
        assert result.year == 2020
        # Should be roughly end of March (366/4 ≈ 91.5 days)
        assert result.month >= 3

class TestSaveDataset:
    @patch('utils.retrieval_utils.cleanup_old_files')
    def test_save_dataset_local(self, mock_cleanup):
        """Test saving dataset locally"""
        df = pd.DataFrame({'col1': [1, 2, 3], 'col2': ['a', 'b', 'c']})
        
        with patch('pandas.DataFrame.to_csv') as mock_to_csv:
            result = save_dataset(df, 'test_dataset', is_local=True)
            
            assert 'test_dataset_' in result
            assert result.endswith('.csv')
            mock_to_csv.assert_called_once()
            mock_cleanup.assert_called_once()

    @patch('boto3.client')
    @patch('utils.retrieval_utils.cleanup_old_files')
    def test_save_dataset_s3(self, mock_cleanup, mock_boto):
        """Test saving dataset to S3"""
        df = pd.DataFrame({'col1': [1, 2, 3], 'col2': ['a', 'b', 'c']})
        mock_s3 = MagicMock()
        mock_boto.return_value = mock_s3
        mock_s3.put_object.return_value = {
            'ResponseMetadata': {'HTTPStatusCode': 200},
            'ETag': 'test-etag'
        }
        
        result = save_dataset(df, 'test_dataset', is_local=False)
        
        assert result.startswith('s3://')
        assert 'test_dataset_' in result
        mock_s3.put_object.assert_called_once()
        mock_cleanup.assert_called_once()

    @patch('boto3.client')
    def test_save_dataset_s3_error(self, mock_boto):
        """Test S3 save error handling"""
        df = pd.DataFrame({'col1': [1, 2, 3]})
        mock_s3 = MagicMock()
        mock_boto.return_value = mock_s3
        mock_s3.put_object.side_effect = Exception("S3 error")
        
        with pytest.raises(Exception):
            save_dataset(df, 'test_dataset', is_local=False)

class TestCleanupOldFiles:
    @patch('glob.glob')
    @patch('pathlib.Path.unlink')
    def test_cleanup_old_files_local(self, mock_unlink, mock_glob):
        """Test cleanup of local files"""
        mock_glob.return_value = [
            '/path/test_20240101.csv',
            '/path/test_20240102.csv',
            '/path/test_20240103.csv'
        ]
        keep_file = '/path/test_20240103.csv'
        
        cleanup_old_files('/path/test_*.csv', keep_file, is_local=True)
        
        # Should attempt to remove 2 old files
        assert mock_unlink.call_count == 2

    @patch('boto3.client')
    def test_cleanup_old_files_s3(self, mock_boto):
        """Test cleanup of S3 files"""
        mock_s3 = MagicMock()
        mock_boto.return_value = mock_s3
        mock_s3.list_objects_v2.return_value = {
            'Contents': [
                {'Key': 'processed/test_20240101.csv'},
                {'Key': 'processed/test_20240102.csv'},
                {'Key': 'processed/test_20240103.csv'}
            ]
        }
        keep_file = 'processed/test_20240103.csv'
        
        cleanup_old_files('processed/test_*.csv', keep_file, is_local=False)
        
        # Should attempt to delete 2 old files
        assert mock_s3.delete_object.call_count == 2

class TestErrorHandling:
    @patch('utils.retrieval_utils.get_file_date')
    def test_needs_update_handles_exceptions(self, mock_get_date):
        """Test that needs_update returns True when exceptions occur"""
        mock_get_date.side_effect = Exception("File system error")
        config = {'test_dataset': {'update_cadence': 'daily'}}
        
        result = needs_update('test_dataset', config, True)
        
        # Should default to True when errors occur
        assert result is True

    @patch('glob.glob')
    def test_get_files_list_handles_exceptions(self, mock_glob):
        """Test that get_files_list handles exceptions gracefully"""
        mock_glob.side_effect = Exception("Permission denied")
        
        result = get_files_list('test_dataset', is_local=True)
        
        # Should return empty list on error
        assert result == []