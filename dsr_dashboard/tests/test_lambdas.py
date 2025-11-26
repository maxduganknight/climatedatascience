import pytest
import json
import os
from unittest.mock import patch, MagicMock
from datetime import datetime
from botocore.exceptions import ClientError

class TestPresignedUrl:
    @patch.dict(os.environ, {'DASHBOARD_METRICS_BUCKET': 'test-bucket'})
    @patch('boto3.client')
    def test_presigned_url_valid_key(self, mock_boto):
        """Test presigned URL generation for valid key"""
        from lambdas.presigned_url import handler
        
        mock_s3 = MagicMock()
        mock_boto.return_value = mock_s3
        mock_s3.generate_presigned_url.return_value = "https://test-bucket.s3.amazonaws.com/processed/test.csv?signature=abc123"
        
        event = {
            'queryStringParameters': {'key': 'test_dataset_20240101.csv'},
            'headers': {'origin': 'https://www.deepskyclimate.com'},
            'httpMethod': 'GET'
        }
        
        response = handler(event, {})
        
        assert response['statusCode'] == 200
        body = json.loads(response['body'])
        assert 'url' in body
        assert 'test-bucket.s3.amazonaws.com' in body['url']
        
        # Verify S3 call
        mock_s3.generate_presigned_url.assert_called_once()
        call_args = mock_s3.generate_presigned_url.call_args
        assert call_args[0][0] == 'get_object'
        assert call_args[1]['Params']['Key'] == 'processed/test_dataset_20240101.csv'

    @patch.dict(os.environ, {'DASHBOARD_METRICS_BUCKET': 'test-bucket'})
    @patch('boto3.client')
    def test_presigned_url_wildcard_pattern(self, mock_boto):
        """Test presigned URL generation with wildcard pattern"""
        from lambdas.presigned_url import handler
        
        mock_s3 = MagicMock()
        mock_boto.return_value = mock_s3
        
        # Mock list_objects_v2 response
        mock_s3.list_objects_v2.return_value = {
            'Contents': [
                {'Key': 'processed/test_dataset_20240101.csv'},
                {'Key': 'processed/test_dataset_20240103.csv'},
                {'Key': 'processed/test_dataset_20240102.csv'}
            ]
        }
        mock_s3.generate_presigned_url.return_value = "https://test-bucket.s3.amazonaws.com/processed/test_dataset_20240103.csv?signature=abc123"
        
        event = {
            'queryStringParameters': {'key': 'test_dataset_*.csv'},
            'headers': {'origin': 'https://www.deepskyclimate.com'},
            'httpMethod': 'GET'
        }
        
        response = handler(event, {})
        
        assert response['statusCode'] == 200
        body = json.loads(response['body'])
        assert 'url' in body
        
        # Should select the latest file (20240103)
        mock_s3.generate_presigned_url.assert_called_once()
        call_args = mock_s3.generate_presigned_url.call_args
        assert call_args[1]['Params']['Key'] == 'processed/test_dataset_20240103.csv'

    @patch.dict(os.environ, {'DASHBOARD_METRICS_BUCKET': 'test-bucket'})
    @patch('boto3.client')
    def test_presigned_url_file_not_found(self, mock_boto):
        """Test handling of file not found"""
        from lambdas.presigned_url import handler
        
        mock_s3 = MagicMock()
        mock_boto.return_value = mock_s3
        
        # Mock ClientError for file not found
        error_response = {'Error': {'Code': 'NoSuchKey'}}
        mock_s3.generate_presigned_url.side_effect = ClientError(error_response, 'get_object')
        
        event = {
            'queryStringParameters': {'key': 'nonexistent_file.csv'},
            'headers': {'origin': 'https://www.deepskyclimate.com'},
            'httpMethod': 'GET'
        }
        
        response = handler(event, {})
        
        assert response['statusCode'] == 404
        body = json.loads(response['body'])
        assert 'error' in body
        assert 'File not found' in body['error']

    @patch.dict(os.environ, {'DASHBOARD_METRICS_BUCKET': 'test-bucket'})
    @patch('boto3.client')
    def test_presigned_url_list_files(self, mock_boto):
        """Test listing available files when no key specified"""
        from lambdas.presigned_url import handler
        
        mock_s3 = MagicMock()
        mock_boto.return_value = mock_s3
        
        # Mock list_objects_v2 response
        mock_s3.list_objects_v2.return_value = {
            'Contents': [
                {'Key': 'processed/dataset1_20240101.csv'},
                {'Key': 'processed/dataset2_20240102.csv'},
                {'Key': 'processed/'}  # Directory entry to be filtered out
            ]
        }
        
        event = {
            'queryStringParameters': {},
            'headers': {'origin': 'https://www.deepskyclimate.com'},
            'httpMethod': 'GET'
        }
        
        response = handler(event, {})
        
        assert response['statusCode'] == 200
        body = json.loads(response['body'])
        assert 'files' in body
        assert len(body['files']) == 2
        assert 'dataset1_20240101.csv' in body['files']
        assert 'dataset2_20240102.csv' in body['files']

    def test_presigned_url_options_request(self):
        """Test handling of OPTIONS preflight request"""
        from lambdas.presigned_url import handler
        
        event = {
            'httpMethod': 'OPTIONS',
            'headers': {'origin': 'https://www.deepskyclimate.com'}
        }
        
        response = handler(event, {})
        
        assert response['statusCode'] == 200
        assert response['body'] == ''
        assert 'Access-Control-Allow-Origin' in response['headers']

    @patch.dict(os.environ, {'DASHBOARD_METRICS_BUCKET': 'test-bucket'})
    @patch('boto3.client')
    def test_presigned_url_viz_path(self, mock_boto):
        """Test handling of viz/ path files"""
        from lambdas.presigned_url import handler
        
        mock_s3 = MagicMock()
        mock_boto.return_value = mock_s3
        mock_s3.generate_presigned_url.return_value = "https://test-bucket.s3.amazonaws.com/viz/chart.js?signature=abc123"
        
        event = {
            'queryStringParameters': {'key': 'viz/chart_templates.js'},
            'headers': {'origin': 'https://www.deepskyclimate.com'},
            'httpMethod': 'GET'
        }
        
        response = handler(event, {})
        
        assert response['statusCode'] == 200
        
        # Should use viz/ path directly without adding processed/ prefix
        call_args = mock_s3.generate_presigned_url.call_args
        assert call_args[1]['Params']['Key'] == 'viz/chart_templates.js'
        assert call_args[1]['Params']['ResponseContentType'] == 'application/javascript'

class TestDatasetVersions:
    @patch.dict(os.environ, {'DASHBOARD_METRICS_BUCKET': 'test-bucket'})
    @patch('boto3.client')
    def test_dataset_versions_success(self, mock_boto):
        """Test successful dataset versions retrieval"""
        from lambdas.dataset_versions import handler
        
        mock_s3 = MagicMock()
        mock_boto.return_value = mock_s3
        
        # Mock S3 response with multiple datasets
        mock_s3.list_objects_v2.return_value = {
            'Contents': [
                {'Key': 'processed/co2_ppm_20240101.csv'},
                {'Key': 'processed/co2_ppm_20240103.csv'},  # Latest for co2_ppm
                {'Key': 'processed/era5_t2m_anom_year_20240102.csv'},
                {'Key': 'processed/aviso_slr_20240104.csv'}
            ]
        }
        
        event = {
            'httpMethod': 'GET',
            'headers': {'origin': 'https://www.deepskyclimate.com'}
        }
        
        response = handler(event, {})
        
        assert response['statusCode'] == 200
        body = json.loads(response['body'])
        
        assert 'versions' in body
        assert 'server_time' in body
        
        # Check that latest versions are selected
        versions = body['versions']
        assert 'co2_ppm_*.csv' in versions
        assert versions['co2_ppm_*.csv'] == '20240103'  # Latest version
        assert 'era5_t2m_anom_year_*.csv' in versions
        assert versions['era5_t2m_anom_year_*.csv'] == '20240102'
        assert 'aviso_slr_*.csv' in versions
        assert versions['aviso_slr_*.csv'] == '20240104'

    @patch.dict(os.environ, {'DASHBOARD_METRICS_BUCKET': 'test-bucket'})
    @patch('boto3.client')
    def test_dataset_versions_empty_bucket(self, mock_boto):
        """Test handling of empty bucket"""
        from lambdas.dataset_versions import handler
        
        mock_s3 = MagicMock()
        mock_boto.return_value = mock_s3
        mock_s3.list_objects_v2.return_value = {}  # Empty bucket
        
        event = {
            'httpMethod': 'GET',
            'headers': {'origin': 'https://www.deepskyclimate.com'}
        }
        
        response = handler(event, {})
        
        assert response['statusCode'] == 200
        body = json.loads(response['body'])
        assert 'versions' in body
        assert body['versions'] == {}

    @patch.dict(os.environ, {})  # No bucket environment variable
    def test_dataset_versions_no_bucket_env(self):
        """Test handling when bucket environment variable is missing"""
        from lambdas.dataset_versions import handler
        
        event = {
            'httpMethod': 'GET',
            'headers': {'origin': 'https://www.deepskyclimate.com'}
        }
        
        response = handler(event, {})
        
        assert response['statusCode'] == 200
        body = json.loads(response['body'])
        assert body['versions'] == {}

    def test_dataset_versions_options_request(self):
        """Test handling of OPTIONS preflight request"""
        from lambdas.dataset_versions import handler
        
        event = {
            'httpMethod': 'OPTIONS',
            'headers': {'origin': 'https://www.deepskyclimate.com'}
        }
        
        response = handler(event, {})
        
        assert response['statusCode'] == 200
        assert response['body'] == ''
        assert 'Access-Control-Allow-Origin' in response['headers']

    @patch.dict(os.environ, {'DASHBOARD_METRICS_BUCKET': 'test-bucket'})
    @patch('boto3.client')
    def test_dataset_versions_s3_error(self, mock_boto):
        """Test handling of S3 errors gracefully returns empty versions"""
        from lambdas.dataset_versions import handler
        
        mock_s3 = MagicMock()
        mock_boto.return_value = mock_s3
        mock_s3.list_objects_v2.side_effect = ClientError(
            {'Error': {'Code': 'AccessDenied'}}, 'list_objects_v2'
        )
        
        event = {
            'httpMethod': 'GET',
            'headers': {'origin': 'https://www.deepskyclimate.com'}
        }
        
        response = handler(event, {})
        
        # Should return 200 with empty versions instead of failing
        assert response['statusCode'] == 200
        body = json.loads(response['body'])
        assert 'versions' in body
        assert body['versions'] == {}

    def test_dataset_versions_cors_headers(self):
        """Test CORS headers are properly set"""
        from lambdas.dataset_versions import handler
        
        # Test with allowed origin
        event = {
            'httpMethod': 'GET',
            'headers': {'origin': 'https://deepskyclimate.com'}
        }
        
        response = handler(event, {})
        
        headers = response['headers']
        assert headers['Access-Control-Allow-Origin'] == 'https://deepskyclimate.com'
        assert headers['Access-Control-Allow-Methods'] == 'GET'
        assert headers['Content-Type'] == 'application/json'

    def test_dataset_versions_invalid_origin(self):
        """Test handling of invalid origin"""
        from lambdas.dataset_versions import handler
        
        event = {
            'httpMethod': 'GET',
            'headers': {'origin': 'https://malicious-site.com'}
        }
        
        response = handler(event, {})
        
        # Should default to main allowed origin
        assert response['headers']['Access-Control-Allow-Origin'] == 'https://www.deepskyclimate.com'

class TestGetDatasetVersions:
    @patch.dict(os.environ, {'DASHBOARD_METRICS_BUCKET': 'test-bucket'})
    @patch('boto3.client')
    def test_get_dataset_versions_function(self, mock_boto):
        """Test the get_dataset_versions helper function"""
        from lambdas.dataset_versions import get_dataset_versions
        
        mock_s3 = MagicMock()
        mock_boto.return_value = mock_s3
        
        mock_s3.list_objects_v2.return_value = {
            'Contents': [
                {'Key': 'processed/dataset1_20240101.csv'},
                {'Key': 'processed/dataset1_20240103.csv'},  # Latest
                {'Key': 'processed/dataset2_20240102.csv'},
                {'Key': 'processed/invalid_filename.csv'},  # Should be ignored
                {'Key': 'processed/config.json'}  # Not a CSV, should be ignored
            ]
        }
        
        result = get_dataset_versions()
        
        assert len(result) == 2
        assert result['dataset1'] == '20240103'  # Latest version selected
        assert result['dataset2'] == '20240102'
        assert 'invalid_filename' not in result

    @patch.dict(os.environ, {})
    def test_get_dataset_versions_no_bucket(self):
        """Test get_dataset_versions when bucket env var is missing"""
        from lambdas.dataset_versions import get_dataset_versions
        
        result = get_dataset_versions()
        
        assert result == {}

    @patch.dict(os.environ, {'DASHBOARD_METRICS_BUCKET': 'test-bucket'})
    @patch('boto3.client')
    def test_get_dataset_versions_boto_error(self, mock_boto):
        """Test get_dataset_versions with boto error"""
        from lambdas.dataset_versions import get_dataset_versions
        
        mock_boto.side_effect = Exception("AWS credentials error")
        
        result = get_dataset_versions()
        
        # Should return empty dict on error, not raise exception
        assert result == {}