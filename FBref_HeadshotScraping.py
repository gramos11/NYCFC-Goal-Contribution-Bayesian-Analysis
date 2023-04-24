# -*- coding: utf-8 -*-
"""
Created on Wed Apr 19 18:18:43 2023

@author: Garrett Ramos
"""

import requests
from bs4 import BeautifulSoup
from PIL import Image
from io import BytesIO


def get_player_url(player_name):
    search_url = "https://www.fbref.com/search/search.fcgi"
    params = {"search": player_name}

    response = requests.get(search_url, params=params)
    current_url = response.url
    html_content = response.text

    soup = BeautifulSoup(html_content, 'html.parser')

    content_div = soup.find("div", {"id": "content"})
    search_results_div = content_div.find("div", {"class": "search-results"})
    if search_results_div is None:
        url = current_url
    else:
        searches_div = search_results_div.find("div", {"id": "searches"})

        player_url = searches_div.find("div", {"class": "search-item-url"}).text.strip()
        url = 'https://www.fbref.com/' + player_url
    
    return url


def get_player_headshot(player_url):
    response = requests.get(player_url)
    html_content = response.text
    soup = BeautifulSoup(html_content, 'html.parser')
    img = soup.find('div', {'id': 'info'})
    img2 = img.find('div', {'id': 'meta'})
    img3 = img2.find('div', {'class': 'media-item'})
    jpg = img3.find('img')
    headshot_url = jpg['src']
    return headshot_url


def save_headshot_jpg(headshot_url, player_url):
    response = requests.get(headshot_url)
    img = Image.open(BytesIO(response.content))
    img.save(f'{player_url.split("/")[-1]}.jpg')
    