# from http import client
from supabase import create_client, Client
import requests 
import json
import google.generativeai as genai
from requests.auth import HTTPBasicAuth

def init_gemini():
    genai.configure(api_key="AIzaSyDNcnPNLS0Qg3Wjan8L-ok3V3pjb-4-1iQ")
    model = genai.GenerativeModel('gemini-2.0-flash')
    return model

def init_supabase():
    VITE_SUPABASE_URL="https://mouwhbulaoghvsxbvmwj.supabase.co"
    VITE_SUPABASE_ANON_KEY="eyJhbGciOiJIUzI1NiIsInR5cCI6IkpXVCJ9.eyJpc3MiOiJzdXBhYmFzZSIsInJlZiI6Im1vdXdoYnVsYW9naHZzeGJ2bXdqIiwicm9sZSI6ImFub24iLCJpYXQiOjE3NDA0NjI1NzQsImV4cCI6MjA1NjAzODU3NH0.52PcqiCjO8L1VU1lY7t01VLVSD_Cvz0OQFuPfT7lJ2w"

    supabase_url = VITE_SUPABASE_URL
    supabase_key = VITE_SUPABASE_ANON_KEY
    supabase: Client = create_client(supabase_url, supabase_key)
    return supabase

def get_article_content(supabase, article_id):
    try:
        response = (supabase.table("articles")
            .select("*")
            .eq("article_id", article_id)
            .execute())
        
        if not response.data or len(response.data) == 0:
            print(f"No article found with ID: {article_id}")
            return ""
            
        article_content = response.data[0]['content_text']
        return article_content
    except Exception as e:
        print(f"Error getting article content: {str(e)}")
        return ""

def get_image_query(client,article_content):
    response = client.generate_content(
        contents=["""Generate a two to three word summary that can be used to query a relevant thumbnail image from the Unsplash API for the given content. The output should be descriptive and focused on visually representing the concept in a way that aligns with  themes when applicable. Avoid literal interpretations that may return unrelated results (e.g., for "support vector machines", avoid words that might return images of doctors). Do not use any formatting or special characters in the output—only return the words.""",article_content])
    return response.text


def get_best_image(client,image_links,article_content):
    response = client.generate_content(
        contents=["""You are given three image URLs and one HTML article or text passage. Analyze the content of the article and determine which image is the most contextually and semantically relevant to the article. Return "0" if the Unsplash image is the best match, "1" if the Pexels image is the best match Return only the number as a string — "0" or "1" — with no additional text or explanation.""",article_content,image_links[0],image_links[1]])
    print(response.text)
    return response.text

    
def query_unsplash(query,num_images):
    access_key = "9l9BYiRN2d3BjXbI4SjFzlnPkGJNBPUIecGKVzyN5b0"  # Replace with your Unsplash API access key
    url = f"https://api.unsplash.com/search/photos?query={query}&per_page={num_images}&client_id={access_key}"
    response = requests.get(url)
    data = json.loads(response.text)
    image_links = [result["urls"]["regular"] for result in data["results"]]
    return image_links[0]

     
def append_link_in_supabase(supabase,article_id,unsplash_link):
    response = (supabase.table("articles")
    .update({"content_img": unsplash_link})
    .eq("article_id", article_id)
    .execute())

    # print(response)
    print("success !!")


def unsplash_api_fetcher(article_id):
    supabase = init_supabase()
    article_content = get_article_content(supabase,article_id)
    client = init_gemini()


    image_links = []



    query = get_image_query(client,article_content)
    # print(article_content)

    unsplash_link = query_unsplash(query,1)
    image_links.append(unsplash_link)
    # print("unsplash_link",unsplash_link)

    # unsplash_link = search_image(query)
    # image_links.append(unsplash_link)
    # print("shutterstock_link",unsplash_link)

    unsplash_link = search_pexels(query)
    image_links.append(unsplash_link)
    # print("pexels_link",unsplash_link)


    index = get_best_image(client,image_links,article_content)
    unsplash_link = image_links[int(index)]

    append_link_in_supabase(supabase,article_id,unsplash_link)


def search_image(query):
    # Your Shutterstock API credentials
    api_key = 'mHB5wurBT22jAjgluyG3pABAAgOLwLut'
    api_secret = 'o2R3VjlPLGfFAtkb'
    
    # Get access token
    token_url = 'https://api.shutterstock.com/v2/oauth/access_token'
    token_data = {
        'grant_type': 'client_credentials',
        'client_id': api_key,
        'client_secret': api_secret
    }
    token_response = requests.post(token_url, data=token_data)
    token_response.raise_for_status()
    access_token = token_response.json().get('access_token')

    # Search for images
    search_url = 'https://api.shutterstock.com/v2/images/search'
    headers = {'Authorization': f'Bearer {access_token}'}
    params = {'query': query, 'per_page': 1}
    search_response = requests.get(search_url, headers=headers, params=params)
    search_response.raise_for_status()
    data = search_response.json()

    if data['data']:
        # print(data)
        return data['data'][0]['assets']['huge_thumb']['url']
    else:
        return 'No images found.'


def search_pexels(query):
    api_key = "hWGudU4S7Mygp8JMCpvSESQPfTDYhN57w3Q6nfeabQ7fetAeAQQyxScf"  
    url = f"https://api.pexels.com/v1/search?query={query}&per_page=1"
    headers = {"Authorization": api_key}
    
    response = requests.get(url, headers=headers)
    response.raise_for_status()
    data = response.json()
    
    if data['photos']:
        return data['photos'][0]['src']['original']  # Returns the large size image URL
    else:
        return 'No images found.'











    
        








