#!/usr/bin/env python3
"""
Script to generate a fresh Gmail token for GitHub Actions.
Run this locally to create a new token.json file, then encode it for GitHub Secrets.
"""

import os
import base64
from google_auth_oauthlib.flow import InstalledAppFlow
from google.auth.transport.requests import Request
from google.oauth2.credentials import Credentials

# Gmail API scopes
SCOPES = ['https://www.googleapis.com/auth/gmail.send']

def generate_token():
    """Generate a fresh Gmail token for GitHub Actions."""
    
    if not os.path.exists('credentials.json'):
        print("❌ Error: credentials.json not found!")
        print("Please download your OAuth 2.0 credentials from Google Cloud Console")
        print("and save them as 'credentials.json' in this directory.")
        return False
    
    try:
        # Create flow and get credentials
        flow = InstalledAppFlow.from_client_secrets_file('credentials.json', SCOPES)
        creds = flow.run_local_server(port=0)
        
        # Save the credentials
        with open('token.json', 'w') as token:
            token.write(creds.to_json())
        
        print("✅ Token generated successfully!")
        print("📄 token.json has been created.")
        
        # Encode for GitHub Secrets
        with open('token.json', 'r') as f:
            token_content = f.read()
        
        token_b64 = base64.b64encode(token_content.encode()).decode()
        
        print("\n🔐 For GitHub Actions, add this to your repository secrets:")
        print("Secret name: TOKEN_JSON_B64")
        print("Secret value:")
        print(token_b64)
        
        return True
        
    except Exception as e:
        print(f"❌ Error generating token: {e}")
        return False

if __name__ == "__main__":
    print("🔑 Gmail Token Generator for GitHub Actions")
    print("=" * 50)
    
    if generate_token():
        print("\n✅ Setup complete! Update your GitHub repository secrets with the TOKEN_JSON_B64 value above.")
    else:
        print("\n❌ Setup failed. Please check your credentials.json file.") 