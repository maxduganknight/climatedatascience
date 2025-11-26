import pytest
import pandas as pd
import requests
from unittest.mock import patch, MagicMock
import os
import warnings
from datetime import datetime

class TestHomeInsurancePremium:
    
    @patch('utils.retrieval_utils.get_aws_secret')
    @patch.dict(os.environ, {'AWS_LAMBDA_FUNCTION_NAME': 'test-lambda'})
    @patch('requests.get')
    def test_download_home_insurance_premium_lambda(self, mock_get, mock_secret):
        """Test home insurance data download in Lambda environment"""
        from retrieval.home_insurance_premium import download_home_insurance_premium
        
        # Mock AWS secret
        mock_secret.return_value = {'FRED_API_KEY': 'test-fred-key'}
        
        # Mock FRED API response
        mock_response = MagicMock()
        mock_response.status_code = 200
        mock_response.raise_for_status.return_value = None
        mock_response.json.return_value = {
            'observations': [
                {'date': '2020-01-01', 'value': '100.0'},
                {'date': '2021-01-01', 'value': '105.0'},
                {'date': '2022-01-01', 'value': '110.25'}
            ]
        }
        mock_get.return_value = mock_response
        
        config = {
            'home_insurance_premium': {
                'data_source_url': 'https://api.stlouisfed.org/fred/series/observations',
                'dataset_id': 'PCU9241269241262'
            }
        }
        
        result = download_home_insurance_premium(config)
        
        # Verify result structure
        assert isinstance(result, pd.DataFrame)
        assert len(result) == 3
        assert all(col in result.columns for col in ['year', 'premium_index', 'percent_increase'])
        
        # Verify data processing
        assert result['percent_increase'].iloc[0] == 0.0  # 100 - 100
        assert result['percent_increase'].iloc[1] == 5.0  # 105 - 100
        assert result['percent_increase'].iloc[2] == 10.25  # 110.25 - 100
        
        # Verify API call
        mock_get.assert_called_once()
        call_args = mock_get.call_args
        assert 'api_key' in call_args[1]['params']
        assert call_args[1]['params']['series_id'] == 'PCU9241269241262'

    @patch.dict(os.environ, {}, clear=True)
    @patch('sys.modules', {'creds': MagicMock(FRED_API_KEY='local-fred-key'), 'warnings': warnings})
    @patch('requests.get')
    def test_download_home_insurance_premium_local(self, mock_get):
        """Test home insurance data download in local environment"""
        from retrieval.home_insurance_premium import download_home_insurance_premium
        
        # Mock successful API response
        mock_response = MagicMock()
        mock_response.status_code = 200
        mock_response.raise_for_status.return_value = None
        mock_response.json.return_value = {
            'observations': [
                {'date': '2020-06-01', 'value': '98.5'},
                {'date': '2020-09-01', 'value': '99.2'},
                {'date': '2020-12-01', 'value': '100.8'}
            ]
        }
        mock_get.return_value = mock_response
        
        config = {
            'home_insurance_premium': {
                'data_source_url': 'https://api.stlouisfed.org/fred/series/observations',
                'dataset_id': 'PCU9241269241262'
            }
        }
        
        result = download_home_insurance_premium(config)
        
        assert isinstance(result, pd.DataFrame)
        assert len(result) == 3
        assert all(isinstance(date, pd.Timestamp) for date in result['year'])

    @patch('sys.modules', {'creds': MagicMock(FRED_API_KEY='test-key'), 'warnings': warnings})
    @patch('requests.get')
    def test_download_home_insurance_premium_api_error(self, mock_get):
        """Test handling of API errors"""
        from retrieval.home_insurance_premium import download_home_insurance_premium
        
        # Mock API error
        mock_get.side_effect = requests.RequestException("API unavailable")
        
        config = {
            'home_insurance_premium': {
                'data_source_url': 'https://api.stlouisfed.org/fred/series/observations',
                'dataset_id': 'PCU9241269241262'
            }
        }
        
        with pytest.raises(requests.RequestException):
            download_home_insurance_premium(config)

    @patch('sys.modules', {'creds': MagicMock(FRED_API_KEY='test-key'), 'warnings': warnings})
    @patch('requests.get')
    def test_download_home_insurance_premium_http_error(self, mock_get):
        """Test handling of HTTP errors"""
        from retrieval.home_insurance_premium import download_home_insurance_premium
        
        # Mock HTTP error response
        mock_response = MagicMock()
        mock_response.raise_for_status.side_effect = requests.HTTPError("404 Not Found")
        mock_get.return_value = mock_response
        
        config = {
            'home_insurance_premium': {
                'data_source_url': 'https://api.stlouisfed.org/fred/series/observations',
                'dataset_id': 'PCU9241269241262'
            }
        }
        
        with pytest.raises(requests.HTTPError):
            download_home_insurance_premium(config)

    @patch('sys.modules', {'creds': MagicMock(FRED_API_KEY='test-key'), 'warnings': warnings})
    @patch('requests.get')
    def test_download_home_insurance_premium_missing_values(self, mock_get):
        """Test handling of missing values in API response"""
        from retrieval.home_insurance_premium import download_home_insurance_premium
        
        # Mock API response with missing values
        mock_response = MagicMock()
        mock_response.status_code = 200
        mock_response.raise_for_status.return_value = None
        mock_response.json.return_value = {
            'observations': [
                {'date': '2020-01-01', 'value': '100.0'},
                {'date': '2021-01-01', 'value': '.'},  # Missing value indicator
                {'date': '2022-01-01', 'value': '110.0'}
            ]
        }
        mock_get.return_value = mock_response
        
        config = {
            'home_insurance_premium': {
                'data_source_url': 'https://api.stlouisfed.org/fred/series/observations',
                'dataset_id': 'PCU9241269241262'
            }
        }
        
        result = download_home_insurance_premium(config)
        
        # Missing values should be dropped, so we should only have 2 rows
        assert len(result) == 2
        assert all(pd.notna(result['percent_increase']))
        # Verify the remaining data is correct
        assert result['percent_increase'].iloc[0] == 0.0  # 100 - 100
        assert result['percent_increase'].iloc[1] == 10.0  # 110 - 100

    @patch('retrieval.home_insurance_premium.save_dataset')  # Mock where it's used
    @patch('retrieval.home_insurance_premium.download_home_insurance_premium')
    def test_process_home_insurance_premium(self, mock_download, mock_save):
        """Test the process_home_insurance_premium function"""
        from retrieval.home_insurance_premium import process_home_insurance_premium
        
        # Mock data
        mock_df = pd.DataFrame({
            'year': pd.to_datetime(['2020-01-01', '2021-01-01']),
            'premium_index': [100.0, 105.0],
            'percent_increase': [0.0, 5.0]
        })
        mock_download.return_value = mock_df
        mock_save.return_value = '/path/to/saved/file.csv'
        
        config = {'test': 'config'}
        result = process_home_insurance_premium(config, is_local=True)
        
        assert result == '/path/to/saved/file.csv'
        mock_download.assert_called_once_with(config)
        mock_save.assert_called_once_with(mock_df, 'home_insurance_premium', True)

class TestArcticSeaIce:
    @patch('requests.get')
    def test_download_arctic_sea_ice_success(self, mock_get):
        """Test successful Arctic sea ice data download"""
        from retrieval.arctic_sea_ice import download_arctic_sea_ice
        
        # Mock CSV response
        mock_response = MagicMock()
        mock_response.status_code = 200
        mock_response.text = """year, extent, area
1979,7.20,6.50
1980,7.83,7.10
1981,7.25,6.58"""
        mock_get.return_value = mock_response
        
        # Mock pandas read_csv to use our test data
        with patch('pandas.read_csv') as mock_read_csv:
            mock_read_csv.return_value = pd.DataFrame({
                'year': [1979, 1980, 1981],
                ' extent': [7.20, 7.83, 7.25],  # Note the space in column name
                'area': [6.50, 7.10, 6.58]
            })
            
            config = {
                'arctic_sea_ice': {
                    'data_source_url': 'https://noaadata.apps.nsidc.org/test.csv'
                }
            }
            
            result = download_arctic_sea_ice(config)
            
            assert isinstance(result, pd.DataFrame)
            assert len(result) == 3
            assert list(result.columns) == ['year', 'extent']
            
            # Verify date conversion (should be September 1st for each year)
            assert all(date.month == 9 and date.day == 1 for date in result['year'])
            assert result['year'].iloc[0].year == 1979
            
            # Verify extent values
            assert result['extent'].iloc[0] == 7.20
            assert result['extent'].iloc[1] == 7.83

    @patch('requests.get')
    def test_download_arctic_sea_ice_http_error(self, mock_get):
        """Test handling of HTTP errors"""
        from retrieval.arctic_sea_ice import download_arctic_sea_ice
        
        # Mock HTTP error
        mock_response = MagicMock()
        mock_response.status_code = 404
        mock_get.return_value = mock_response
        
        config = {
            'arctic_sea_ice': {
                'data_source_url': 'https://noaadata.apps.nsidc.org/test.csv'
            }
        }
        
        with pytest.raises(RuntimeError, match="Failed to download data"):
            download_arctic_sea_ice(config)

    @patch('requests.get')
    def test_download_arctic_sea_ice_network_error(self, mock_get):
        """Test handling of network errors"""
        from retrieval.arctic_sea_ice import download_arctic_sea_ice
        
        mock_get.side_effect = requests.RequestException("Network error")
        
        config = {
            'arctic_sea_ice': {
                'data_source_url': 'https://noaadata.apps.nsidc.org/test.csv'
            }
        }
        
        with pytest.raises(requests.RequestException):
            download_arctic_sea_ice(config)

    @patch('pandas.read_csv')
    @patch('requests.get')
    def test_download_arctic_sea_ice_data_processing_error(self, mock_get, mock_read_csv):
        """Test handling of data processing errors"""
        from retrieval.arctic_sea_ice import download_arctic_sea_ice
        
        mock_response = MagicMock()
        mock_response.status_code = 200
        mock_get.return_value = mock_response
        
        # Mock pandas error
        mock_read_csv.side_effect = pd.errors.ParserError("Unable to parse CSV")
        
        config = {
            'arctic_sea_ice': {
                'data_source_url': 'https://noaadata.apps.nsidc.org/test.csv'
            }
        }
        
        with pytest.raises(pd.errors.ParserError):
            download_arctic_sea_ice(config)

    @patch('retrieval.arctic_sea_ice.save_dataset')  # Mock where it's used
    @patch('retrieval.arctic_sea_ice.download_arctic_sea_ice')
    def test_process_arctic_sea_ice(self, mock_download, mock_save):
        """Test the process_arctic_sea_ice function"""
        from retrieval.arctic_sea_ice import process_arctic_sea_ice
        
        # Mock data
        mock_df = pd.DataFrame({
            'year': pd.to_datetime(['1979-09-01', '1980-09-01']),
            'extent': [7.20, 7.83]
        })
        mock_download.return_value = mock_df
        mock_save.return_value = 's3://test-bucket/arctic_sea_ice_20240101.csv'
        
        config = {'test': 'config'}
        result = process_arctic_sea_ice(config, is_local=False)
        
        assert result == 's3://test-bucket/arctic_sea_ice_20240101.csv'
        mock_download.assert_called_once_with(config)
        mock_save.assert_called_once_with(mock_df, 'arctic_sea_ice', False)

    @patch('requests.get')
    def test_download_arctic_sea_ice_empty_data(self, mock_get):
        """Test handling of empty data response"""
        from retrieval.arctic_sea_ice import download_arctic_sea_ice
        
        mock_response = MagicMock()
        mock_response.status_code = 200
        mock_get.return_value = mock_response
        
        # Mock empty DataFrame
        with patch('pandas.read_csv') as mock_read_csv:
            mock_read_csv.return_value = pd.DataFrame()
            
            config = {
                'arctic_sea_ice': {
                    'data_source_url': 'https://noaadata.apps.nsidc.org/test.csv'
                }
            }
            
            with pytest.raises(KeyError):  # Will fail when trying to access columns
                download_arctic_sea_ice(config)