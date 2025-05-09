import os
import sys
import argparse
from dotenv import load_dotenv

# Load environment variables
load_dotenv()

def run_streamlit():
    """Run the Streamlit web application."""
    print("Starting Streamlit web application...")
    os.system("streamlit run app/streamlit_app.py")

def run_telegram_bot():
    """Run the Telegram bot."""
    if not os.getenv("TELEGRAM_BOT_TOKEN"):
        print("Error: TELEGRAM_BOT_TOKEN environment variable is not set.")
        print("Please set it in your .env file or as an environment variable.")
        sys.exit(1)
        
    print("Starting Telegram bot...")
    from app.telegram_bot import main
    main()

def main():
    """Main entry point for the application."""
    parser = argparse.ArgumentParser(description="DeepSeek-Coder RAG Application")
    
    # Add arguments
    parser.add_argument(
        "app", 
        choices=["streamlit", "telegram"],
        help="Application to run: streamlit for web UI or telegram for Telegram bot"
    )
    
    args = parser.parse_args()
    
    if args.app == "streamlit":
        run_streamlit()
    elif args.app == "telegram":
        run_telegram_bot()
    else:
        print(f"Unknown application: {args.app}")
        sys.exit(1)

if __name__ == "__main__":
    main()