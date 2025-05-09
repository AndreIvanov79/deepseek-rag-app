import os
import asyncio
import logging
import tempfile
import zipfile
from pathlib import Path
from typing import Dict, Any, List, Optional

import nest_asyncio
from telegram import Update, InlineKeyboardButton, InlineKeyboardMarkup
from telegram.ext import Application, CommandHandler, MessageHandler, filters, CallbackQueryHandler, ContextTypes

import sys
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from utils.model_loader import ModelLoader
from utils.rag_utils import RAGManager
from utils.chunking import ResponseChunker
from utils.image_utils import CodeImageProcessor

# Enable nested event loops for Jupyter notebooks if needed
nest_asyncio.apply()

# Set up logging
logging.basicConfig(
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s", level=logging.INFO
)
logger = logging.getLogger(__name__)

# Environment setup
TOKEN = os.getenv("TELEGRAM_BOT_TOKEN")
if not TOKEN:
    raise ValueError("No Telegram bot token provided. Set the TELEGRAM_BOT_TOKEN environment variable.")

# Global variables
model_loader = None
rag_manager = None
image_processor = None
user_data = {}  # Store user session data

# User data structure:
# user_data = {
#     user_id: {
#         "selected_project": project_id,
#         "waiting_for_project_name": bool,
#         "temp_zip_path": str,
#         "message_chunks": [list of chunked messages waiting to be sent],
#         "waiting_for_code_action": bool,
#         "temp_image_path": str,
#         "extracted_code": {text, language}
#     }
# }

async def init_model():
    """Initialize the model and RAG manager."""
    global model_loader, rag_manager, image_processor
    
    if model_loader is None:
        logger.info("Initializing DeepSeek-Coder model...")
        model_loader = ModelLoader()
        model_loader.load_model()
        logger.info("Model loaded successfully")
    
    if rag_manager is None:
        logger.info("Initializing RAG manager...")
        rag_manager = RAGManager(model_loader)
        logger.info("RAG manager initialized")
        
    if image_processor is None:
        logger.info("Initializing code image processor...")
        image_processor = CodeImageProcessor()
        logger.info("Code image processor initialized")

async def start_command(update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:
    """Send a message when the command /start is issued."""
    await update.message.reply_text(
        "🧠 Welcome to DeepSeek-Coder Assistant! 🧠\n\n"
        "I can help you with coding queries and leverage your project context for better assistance.\n\n"
        "Commands:\n"
        "/start - Show this help message\n"
        "/projects - Manage your projects\n"
        "/select - Select a project for context\n"
        "/clear - Clear project context\n\n"
        "Upload a ZIP file of your project to create a context database."
    )

async def help_command(update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:
    """Send a message when the command /help is issued."""
    await update.message.reply_text(
        "How to use this bot:\n\n"
        "1. Upload a ZIP file of your project\n"
        "2. Give it a name when prompted\n"
        "3. Select the project with /select command\n"
        "4. Ask coding questions related to your project\n"
        "5. Send photos of code to analyze or extract\n\n"
        "The bot will use the project context to provide more relevant answers."
    )

async def projects_command(update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:
    """List available projects or provide project management options."""
    await init_model()
    
    projects = rag_manager.get_projects()
    
    if not projects:
        await update.message.reply_text(
            "No projects found. Upload a ZIP file to create a new project."
        )
        return
    
    # Create keyboard with project options
    keyboard = []
    for project_id, project_name in projects.items():
        keyboard.append([
            InlineKeyboardButton(f"📂 {project_name}", callback_data=f"select_{project_id}"),
            InlineKeyboardButton("❌ Delete", callback_data=f"delete_{project_id}")
        ])
    
    reply_markup = InlineKeyboardMarkup(keyboard)
    
    await update.message.reply_text(
        "📚 Your Projects:\n\n"
        "Select a project to use as context or delete it:",
        reply_markup=reply_markup
    )

async def select_command(update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:
    """Command to select a project."""
    await init_model()
    
    projects = rag_manager.get_projects()
    
    if not projects:
        await update.message.reply_text(
            "No projects found. Upload a ZIP file to create a new project."
        )
        return
    
    # Create keyboard with project selection options
    keyboard = []
    for project_id, project_name in projects.items():
        keyboard.append([InlineKeyboardButton(f"📂 {project_name}", callback_data=f"select_{project_id}")])
    
    reply_markup = InlineKeyboardMarkup(keyboard)
    
    await update.message.reply_text(
        "Select a project to use as context:",
        reply_markup=reply_markup
    )

async def clear_command(update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:
    """Clear selected project."""
    user_id = update.effective_user.id
    
    if user_id in user_data and user_data[user_id].get("selected_project"):
        user_data[user_id]["selected_project"] = None
        await update.message.reply_text("Project context cleared. I'll respond based on my general knowledge.")
    else:
        await update.message.reply_text("No project is selected.")

async def button_callback(update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:
    """Handle button callbacks from inline keyboards."""
    query = update.callback_query
    await query.answer()
    
    user_id = update.effective_user.id
    callback_data = query.data
    
    # Handle code image action buttons
    if callback_data.startswith("code_"):
        if user_id in user_data and user_data[user_id].get("waiting_for_code_action") and user_data[user_id].get("extracted_code"):
            extracted_code = user_data[user_id]["extracted_code"]
            
            if callback_data == "code_send":
                # Just send the code to the chat
                code_text = f"I've extracted this code from your image:\n```{extracted_code['language']}\n{extracted_code['text']}\n```"
                await query.edit_message_text(code_text)
                
                # Clear the code action state
                user_data[user_id]["waiting_for_code_action"] = False
                user_data[user_id]["extracted_code"] = None
                
            elif callback_data == "code_analyze":
                # Analyze the code with the model
                query_message = f"Please analyze and explain this code:\n```{extracted_code['language']}\n{extracted_code['text']}\n```"
                
                await query.edit_message_text("⏳ Analyzing code... This may take a moment.")
                
                try:
                    await init_model()
                    
                    if user_data[user_id].get("selected_project"):
                        # Use project context for response
                        response_data = rag_manager.query_project(
                            user_data[user_id]["selected_project"], query_message
                        )
                        response_text = response_data["response"]
                    else:
                        # Basic prompt for the model without context
                        model, tokenizer = model_loader.get_model_and_tokenizer()
                        
                        # DeepSeek Chat Template
                        messages = [
                            {"role": "user", "content": query_message}
                        ]
                        chat_text = tokenizer.apply_chat_template(
                            messages, 
                            tokenize=False, 
                            add_generation_prompt=True
                        )
                        
                        # Generate response
                        inputs = tokenizer(chat_text, return_tensors="pt").to(model_loader.device)
                        outputs = model.generate(
                            inputs["input_ids"],
                            max_new_tokens=2048,
                            temperature=0.1,
                            top_p=0.95,
                            do_sample=True
                        )
                        response_text = tokenizer.decode(outputs[0][inputs["input_ids"].shape[1]:], skip_special_tokens=True)
                    
                    # Send the response with chunking if needed
                    await send_chunked_response(update, context, response_text)
                    
                except Exception as e:
                    logger.error(f"Error analyzing code: {str(e)}")
                    await query.message.reply_text(
                        f"❌ Sorry, an error occurred while analyzing the code: {str(e)}"
                    )
                
                # Clear the code action state
                user_data[user_id]["waiting_for_code_action"] = False
                user_data[user_id]["extracted_code"] = None
                
        return
    
    # Handle project selection buttons
    if callback_data.startswith("select_"):
        project_id = callback_data.replace("select_", "")
        projects = rag_manager.get_projects()
        
        if project_id in projects:
            # Initialize user data if not present
            if user_id not in user_data:
                user_data[user_id] = {}
            
            user_data[user_id]["selected_project"] = project_id
            
            await query.edit_message_text(
                f"✅ Project '{projects[project_id]}' selected as context.\n\n"
                f"You can now ask questions related to this project!"
            )
        else:
            await query.edit_message_text("Error: Project not found.")
    
    elif callback_data.startswith("delete_"):
        project_id = callback_data.replace("delete_", "")
        projects = rag_manager.get_projects()
        
        if project_id in projects:
            project_name = projects[project_id]
            
            # Delete the project
            if rag_manager.delete_project(project_id):
                # Clear selection if this project was selected
                if user_id in user_data and user_data[user_id].get("selected_project") == project_id:
                    user_data[user_id]["selected_project"] = None
                
                await query.edit_message_text(f"✅ Project '{project_name}' deleted.")
            else:
                await query.edit_message_text(f"❌ Failed to delete project '{project_name}'.")
        else:
            await query.edit_message_text("Error: Project not found.")

async def process_zip(update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:
    """Process uploaded ZIP file."""
    user_id = update.effective_user.id
    
    await init_model()
    
    # Download the file
    file = await context.bot.get_file(update.message.document.file_id)
    
    with tempfile.NamedTemporaryFile(delete=False, suffix=".zip") as temp_file:
        await file.download_to_drive(custom_path=temp_file.name)
        
        # Check if it's a valid zip file
        try:
            with zipfile.ZipFile(temp_file.name, 'r') as zip_ref:
                # Just test the zip file, don't extract yet
                pass
                
            # Initialize user data if not present
            if user_id not in user_data:
                user_data[user_id] = {}
            
            # Store zip path and wait for project name
            user_data[user_id]["temp_zip_path"] = temp_file.name
            user_data[user_id]["waiting_for_project_name"] = True
            
            await update.message.reply_text(
                "ZIP file received! Please provide a name for this project:"
            )
            
        except zipfile.BadZipFile:
            os.unlink(temp_file.name)
            await update.message.reply_text(
                "❌ The file you uploaded is not a valid ZIP file. Please try again."
            )

async def process_photo(update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:
    """Process a photo containing code."""
    user_id = update.effective_user.id
    
    # Initialize user data if not present
    if user_id not in user_data:
        user_data[user_id] = {}
    
    # Initialize image processor if needed
    global image_processor
    if image_processor is None:
        image_processor = CodeImageProcessor()
    
    # Get the photo with largest size
    photo = update.message.photo[-1]
    
    # Download the photo
    file = await context.bot.get_file(photo.file_id)
    
    with tempfile.NamedTemporaryFile(delete=False, suffix=".jpg") as temp_file:
        await file.download_to_drive(custom_path=temp_file.name)
        image_path = temp_file.name
        
        # Send a status message
        status_message = await update.message.reply_text("🔍 Processing image...")
        
        # Process the image to extract code
        result = image_processor.extract_code_from_image(image_path)
        
        if result["success"]:
            extracted_code = result["text"]
            language = result["detected_language"]
            line_count = result["line_count"]
            
            # Store the extracted code info
            user_data[user_id]["waiting_for_code_action"] = True
            user_data[user_id]["temp_image_path"] = image_path
            user_data[user_id]["extracted_code"] = {
                "text": extracted_code,
                "language": language
            }
            
            # Create keyboard with options
            keyboard = [
                [
                    InlineKeyboardButton("📋 Send Code", callback_data="code_send"),
                    InlineKeyboardButton("🔍 Analyze Code", callback_data="code_analyze")
                ]
            ]
            reply_markup = InlineKeyboardMarkup(keyboard)
            
            # Send a preview of the extracted code
            code_preview = extracted_code
            if len(code_preview) > 200:
                code_preview = code_preview[:197] + "..."
                
            await status_message.edit_text(
                f"✅ Extracted {line_count} lines of code!\n"
                f"Detected language: {language}\n\n"
                f"```\n{code_preview}\n```\n\n"
                f"What would you like to do with this code?",
                reply_markup=reply_markup,
                parse_mode="Markdown"
            )
        else:
            # Clean up temp file on error
            try:
                os.unlink(image_path)
            except:
                pass
                
            await status_message.edit_text(
                f"❌ Failed to extract code from the image: {result.get('error', 'No text detected')}\n\n"
                f"Please make sure the image contains clear, readable code."
            )

async def handle_project_name(update: Update, context: ContextTypes.DEFAULT_TYPE) -> bool:
    """Handle project name input after ZIP upload."""
    user_id = update.effective_user.id
    
    if user_id in user_data and user_data[user_id].get("waiting_for_project_name"):
        project_name = update.message.text.strip()
        temp_zip_path = user_data[user_id]["temp_zip_path"]
        
        # Clear waiting flag
        user_data[user_id]["waiting_for_project_name"] = False
        
        status_message = await update.message.reply_text("⏳ Processing project...")
        
        try:
            # Create index from the zip file
            project_id, doc_count = rag_manager.create_index_from_zip(
                project_name, temp_zip_path
            )
            
            # Clean up the temporary file
            os.unlink(temp_zip_path)
            
            # Set as selected project
            user_data[user_id]["selected_project"] = project_id
            
            await status_message.edit_text(
                f"✅ Project '{project_name}' processed successfully!\n"
                f"Indexed {doc_count} documents.\n\n"
                f"Now using this project for context in our conversation."
            )
            
        except Exception as e:
            # Clean up the temporary file
            if os.path.exists(temp_zip_path):
                os.unlink(temp_zip_path)
                
            await status_message.edit_text(
                f"❌ Error processing project: {str(e)}\n\n"
                f"Please try again with a different ZIP file."
            )
        
        return True
    
    return False

async def send_chunked_response(update: Update, context: ContextTypes.DEFAULT_TYPE, text: str) -> None:
    """Send a large response in chunks if needed."""
    user_id = update.effective_user.id
    
    # Initialize user data if not present
    if user_id not in user_data:
        user_data[user_id] = {}
    
    # Chunk the response if it's too long
    chunks = ResponseChunker.chunk_text(text, max_tokens_per_chunk=1000)
    
    # If we have multiple chunks, indicate it
    if len(chunks) > 1:
        for i, chunk in enumerate(chunks):
            await update.message.reply_text(
                f"Response part {i+1}/{len(chunks)}:\n\n{chunk}"
            )
    else:
        await update.message.reply_text(text)

async def handle_message(update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:
    """Process user messages."""
    # Check if we're waiting for a project name
    if await handle_project_name(update, context):
        return
    
    # Initialize model if needed
    await init_model()
    
    user_id = update.effective_user.id
    query = update.message.text
    
    # Get selected project if any
    selected_project = None
    if user_id in user_data:
        selected_project = user_data[user_id].get("selected_project")
    
    # Send typing action
    await context.bot.send_chat_action(
        chat_id=update.effective_chat.id, 
        action="typing"
    )
    
    try:
        if selected_project:
            # Use project context for response
            response_data = rag_manager.query_project(
                selected_project, query
            )
            response_text = response_data["response"]
        else:
            # Basic prompt for the model without context
            model, tokenizer = model_loader.get_model_and_tokenizer()
            
            # DeepSeek Chat Template
            messages = [
                {"role": "user", "content": query}
            ]
            chat_text = tokenizer.apply_chat_template(
                messages, 
                tokenize=False, 
                add_generation_prompt=True
            )
            
            # Generate response
            inputs = tokenizer(chat_text, return_tensors="pt").to(model_loader.device)
            outputs = model.generate(
                inputs["input_ids"],
                max_new_tokens=2048,
                temperature=0.1,
                top_p=0.95,
                do_sample=True
            )
            response_text = tokenizer.decode(outputs[0][inputs["input_ids"].shape[1]:], skip_special_tokens=True)
        
        # Send the response with chunking if needed
        await send_chunked_response(update, context, response_text)
        
    except Exception as e:
        logger.error(f"Error generating response: {str(e)}")
        await update.message.reply_text(
            f"❌ Sorry, an error occurred while generating the response: {str(e)}"
        )

def main() -> None:
    """Set up and run the Telegram bot."""
    # Create necessary directories
    os.makedirs("./data", exist_ok=True)
    os.makedirs("./data/indices", exist_ok=True)
    os.makedirs("./models", exist_ok=True)
    
    # Create the Application and pass it your bot's token
    application = Application.builder().token(TOKEN).build()
    
    # Add command handlers
    application.add_handler(CommandHandler("start", start_command))
    application.add_handler(CommandHandler("help", help_command))
    application.add_handler(CommandHandler("projects", projects_command))
    application.add_handler(CommandHandler("select", select_command))
    application.add_handler(CommandHandler("clear", clear_command))
    
    # Add callback query handler for inline buttons
    application.add_handler(CallbackQueryHandler(button_callback))
    
    # Add handler for ZIP files
    application.add_handler(MessageHandler(
        filters.Document.ZIP, process_zip
    ))
    
    # Add handler for photos
    application.add_handler(MessageHandler(
        filters.PHOTO, process_photo
    ))
    
    # Add message handler
    application.add_handler(MessageHandler(
        filters.TEXT & ~filters.COMMAND, handle_message
    ))
    
    # Run the bot until the user presses Ctrl-C
    application.run_polling(allowed_updates=Update.ALL_TYPES)

if __name__ == "__main__":
    main()