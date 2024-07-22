import requests
import pandas as pd
from bs4 import BeautifulSoup
import re
import time


def clean_value(value):
    # Remove currency symbols
    value = value.replace('€', '')

    # Check if the value contains 'bn', 'm', or 'k' and convert to millions
    if 'bn' in value:
        num_value = float(value.replace('bn', '')) * 1000
    elif 'm' in value:
        num_value = float(value.replace('m', ''))
    elif 'k' in value:
        num_value = float(value.replace('k', '')) / 1000
    else:
        num_value = float(value)

    return num_value


def scrape_league_data(league_name, transfermarkt_name, year, retries=3, delay=5):
    # Construct the URL based on the league and year
    url = f'https://www.transfermarkt.com/{transfermarkt_name[0]}/startseite/wettbewerb/' \
          f'{transfermarkt_name[1]}/plus/?saison_id={year}'

    # Define headers to mimic a browser request
    headers = {
        'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/91.0.4472.124 Safari/537.36',
        'Accept-Language': 'en-US,en;q=0.9',
        'Accept-Encoding': 'gzip, deflate, br',
        'Accept': 'text/html,application/xhtml+xml,application/xml;q=0.9,image/webp,image/apng,*/*;q=0.8',
        'Connection': 'keep-alive',
        'Upgrade-Insecure-Requests': '1'
    }

    attempt = 0
    while attempt < retries:
        try:
            # Send a GET request to the webpage with a timeout
            response = requests.get(url, headers=headers, timeout=30)
            # Check if the request was successful
            response.raise_for_status()

            # Parse the content of the webpage
            soup = BeautifulSoup(response.content, 'html.parser')

            # Find the table containing the data
            table = soup.find('table', {'class': 'items'})

            if table:
                # Extract table headers
                headers = [header.text.strip() for header in table.find_all('th')]

                # Extract table rows
                rows = []
                for row in table.find_all('tr')[2:]:  # Skipping the first two rows: header row and average row
                    cells = row.find_all('td')
                    if len(cells) > 1:
                        row_data = [cell.text.strip() for cell in cells]
                        rows.append(row_data)

                # Create a DataFrame from the extracted data
                df = pd.DataFrame(rows, columns=headers)

                # Clean and convert the 'ø market value' and 'Total market value' columns
                df['ø market value'] = df['ø market value'].apply(clean_value)
                df['Total market value'] = df['Total market value'].apply(clean_value)

                # Add the 'year' column
                df['year'] = year
                df['league'] = league_name

                df.drop(['Club'], axis=1, inplace=True)

                df.columns = ['team_name', 'team_size', 'mean_age', 'foreigners', 'mean_value', 'total_value', 'year',
                              'league']
                df = df[
                    ['year', 'league', 'team_name', 'team_size', 'mean_age', 'foreigners', 'mean_value', 'total_value']]

                return df

            else:
                print(f"Failed to find the table on the webpage for {league_name} in {year}.")
                return pd.DataFrame()

        except requests.exceptions.RequestException as e:
            print(f"Request failed for {league_name} in {year}: {e}. Retrying in {delay} seconds...")
            attempt += 1
            time.sleep(delay)

    print(f"Failed to retrieve data for {league_name} in {year} after {retries} attempts.")
    return pd.DataFrame()

# Example usage
# league = 'bundesliga'
# year = 2022
# df = scrape_league_data(league, year)
# print(df)

# Save DataFrame to CSV (optional)
# df.to_csv(f'{league}_table_{year}.csv', index=False)
