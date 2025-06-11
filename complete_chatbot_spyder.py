#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
FIXED AI CHATBOT PROJECT - VS CODE COMPATIBLE
============================================
A fully functional AI chatbot with neural networks and multiple interfaces.
All issues fixed for VS Code and Windows environments.

Instructions:
1. Save this file as 'fixed_chatbot_final.py'
2. Run it in VS Code (F5) or terminal: python fixed_chatbot_final.py
3. Choose your interface and start chatting!

Author: AI Assistant
Date: 2025
"""

# =============================================================================
# IMPORTS AND SETUP
# =============================================================================

print("AI CHATBOT LOADING...")
print("=" * 50)

# Check if dependencies are installed
try:
    # Core imports
    import numpy as np
    import json
    import pickle
    import random
    import os
    import sys
    from datetime import datetime
    print("Core libraries loaded")
    
    # Machine Learning imports
    import tensorflow as tf
    from tensorflow.keras.models import Sequential, load_model
    from tensorflow.keras.layers import Dense, Dropout
    from tensorflow.keras.optimizers.legacy import SGD  # Fixed: Use legacy SGD
    print("TensorFlow/Keras loaded")
    
    # NLP imports
    import nltk
    from nltk.stem import WordNetLemmatizer
    print("NLTK loaded")
    
    # GUI imports
    import tkinter as tk
    from tkinter import *
    from tkinter import scrolledtext, messagebox
    print("Tkinter GUI loaded")
    
    # Web framework imports
    try:
        import flask
        from flask import Flask, render_template, request, jsonify
        WEB_AVAILABLE = True
        print("Flask web framework loaded")
    except ImportError:
        WEB_AVAILABLE = False
        print("Flask not available (web interface disabled)")
    
    # Plotting imports
    try:
        import matplotlib.pyplot as plt
        PLOTTING_AVAILABLE = True
        print("Matplotlib plotting loaded")
    except ImportError:
        PLOTTING_AVAILABLE = False
        print("Matplotlib not available (plotting disabled)")

except ImportError as e:
    print(f"Import error: {e}")
    print("Please install missing packages!")
    sys.exit(1)

# Initialize NLTK
try:
    lemmatizer = WordNetLemmatizer()
    # Download NLTK data if needed
    try:
        nltk.data.find('tokenizers/punkt')
        nltk.data.find('corpora/wordnet')
    except LookupError:
        print("Downloading NLTK data...")
        nltk.download('punkt', quiet=True)
        nltk.download('wordnet', quiet=True)
        nltk.download('omw-1.4', quiet=True)
    print("NLTK initialized")
except Exception as e:
    print(f"NLTK initialization failed: {e}")

# Suppress TensorFlow warnings for cleaner output
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'
tf.get_logger().setLevel('ERROR')

print("All systems ready!")
print("=" * 50)

# =============================================================================
# SAMPLE DATA CREATION
# =============================================================================

def create_sample_intents():
    """Create sample training data for the chatbot"""
    intents_data = {
        "intents": [
            {
                "tag": "greeting",
                "patterns": [
                    "Hi", "Hello", "Hey", "Good day", "Greetings", "What's up?", 
                    "How is it going?", "Hi there", "Hey there", "Good morning",
                    "Good afternoon", "Good evening", "Howdy", "Hiya"
                ],
                "responses": [
                    "Hello! How can I help you today?", "Hi there! What can I do for you?", 
                    "Greetings! How may I assist you?", "Hey! Nice to meet you!",
                    "Hello! I'm here to help. What do you need?"
                ]
            },
            {
                "tag": "goodbye",
                "patterns": [
                    "Bye", "See you later", "Goodbye", "Take care", "See you soon", 
                    "Talk to you later", "Catch you later", "Until next time",
                    "Have a great day", "Farewell", "Adios", "Ciao"
                ],
                "responses": [
                    "Goodbye! Have a great day!", "See you later!", "Take care!", 
                    "Until next time!", "Bye! Come back soon!", "Farewell! Stay safe!"
                ]
            },
            {
                "tag": "thanks",
                "patterns": [
                    "Thanks", "Thank you", "That's helpful", "Thank's a lot!", 
                    "Thanks for helping", "Much appreciated", "Thanks a bunch",
                    "I appreciate it", "Thank you so much", "Grateful"
                ],
                "responses": [
                    "You're welcome!", "Happy to help!", "My pleasure!", 
                    "Glad I could assist you!", "Anytime!", "You're very welcome!",
                    "It was my pleasure to help!"
                ]
            },
            {
                "tag": "about",
                "patterns": [
                    "What can you do?", "Who are you?", "What are you?", 
                    "What is your purpose?", "Tell me about yourself",
                    "What are your capabilities?", "How can you help me?",
                    "What do you do?"
                ],
                "responses": [
                    "I'm an AI chatbot created to assist you with various questions and tasks!",
                    "I'm a friendly AI assistant here to help with your questions!",
                    "I'm an intelligent chatbot powered by neural networks. I can chat, answer questions, and help you!",
                    "I'm your AI companion, ready to assist with information and conversation!"
                ]
            },
            {
                "tag": "weather",
                "patterns": [
                    "What's the weather like?", "How's the weather?", "Is it raining?", 
                    "Will it rain today?", "Weather forecast", "Is it sunny?",
                    "What's the temperature?", "Is it cold?", "Is it hot?"
                ],
                "responses": [
                    "I can't check real-time weather, but I recommend checking a weather app or website!",
                    "For current weather, please check your local weather service!",
                    "I don't have access to live weather data, but weather.com or your phone's weather app can help!",
                    "Try asking a weather service for the most up-to-date information!"
                ]
            },
            {
                "tag": "time",
                "patterns": [
                    "What time is it?", "Current time", "What's the time?", 
                    "Time please", "Can you tell me the time?", "What's the current time?"
                ],
                "responses": [
                    "I don't have access to real-time data, please check your device's clock!",
                    "For current time, check your system clock!",
                    "I can't tell time, but your device can show you the current time!",
                    "Check your computer or phone for the current time!"
                ]
            },
            {
                "tag": "help",
                "patterns": [
                    "Help", "I need help", "Can you help me?", "What should I do?", 
                    "Help me", "Support", "Assistance", "I'm stuck", "I need assistance"
                ],
                "responses": [
                    "I'm here to help! What do you need assistance with?",
                    "Of course! How can I assist you?",
                    "I'd be happy to help! What's your question?",
                    "Sure thing! Tell me what you need help with.",
                    "I'm ready to assist! What can I do for you?"
                ]
            },
            {
                "tag": "joke",
                "patterns": [
                    "Tell me a joke", "Make me laugh", "Say something funny", 
                    "Do you know any jokes?", "Joke please", "Tell me something funny",
                    "Make me smile", "I need a laugh"
                ],
                "responses": [
                    "Why don't scientists trust atoms? Because they make up everything!",
                    "I told my wife she was drawing her eyebrows too high. She looked surprised!",
                    "Why do programmers prefer dark mode? Because light attracts bugs!",
                    "What do you call a bear with no teeth? A gummy bear!",
                    "Why don't eggs tell jokes? They'd crack each other up!"
                ]
            },
            {
                "tag": "name",
                "patterns": [
                    "What's your name?", "What should I call you?", "Do you have a name?", 
                    "Who are you?", "What are you called?", "Your name?"
                ],
                "responses": [
                    "You can call me ChatBot!", "I'm your friendly AI assistant!",
                    "I'm ChatBot, nice to meet you!", "I go by ChatBot, but you can call me whatever you like!",
                    "My name is ChatBot, and I'm here to help!"
                ]
            },
            {
                "tag": "age",
                "patterns": [
                    "How old are you?", "What's your age?", "When were you created?", 
                    "Are you old?", "When were you born?", "Your age?"
                ],
                "responses": [
                    "I'm timeless! I exist in the digital realm.",
                    "Age is just a number for AI like me!",
                    "I was just created, so I'm pretty new!",
                    "I don't age like humans do - I'm always learning and updating!",
                    "I'm as old as the code that created me, but my knowledge is constantly growing!"
                ]
            },
            {
                "tag": "compliment",
                "patterns": [
                    "You're awesome", "You're great", "You're helpful", "You're smart",
                    "Good job", "Well done", "You're amazing", "I like you",
                    "You're the best", "You're wonderful"
                ],
                "responses": [
                    "Thank you! That means a lot to me!",
                    "You're very kind! I appreciate the compliment!",
                    "Aww, thank you! You're pretty awesome yourself!",
                    "That's so nice of you to say! I'm happy I could help!",
                    "Thanks! I try my best to be helpful!"
                ]
            }
        ]
    }
    return intents_data

# Create intents data
print("Creating training data...")
intents = create_sample_intents()

# Save intents to file (Fixed: Use UTF-8 encoding)
with open('intents.json', 'w', encoding='utf-8') as file:
    json.dump(intents, file, indent=4)
print("Training data created and saved!")

# =============================================================================
# DATA PREPROCESSING
# =============================================================================

def preprocess_data(intents_data):
    """Preprocess the training data for the neural network"""
    print("Preprocessing training data...")
    
    words = []
    classes = []
    documents = []
    ignore_letters = ['!', '?', ',', '.', "'", '"', ';', ':', '-', '(', ')']
    
    # Process each intent
    for intent in intents_data['intents']:
        for pattern in intent['patterns']:
            # Tokenize each word in the pattern
            word_list = nltk.word_tokenize(pattern)
            words.extend(word_list)
            
            # Add to documents
            documents.append((word_list, intent['tag']))
            
            # Add to classes list
            if intent['tag'] not in classes:
                classes.append(intent['tag'])
    
    # Lemmatize and clean words
    words = [lemmatizer.lemmatize(word.lower()) for word in words if word not in ignore_letters]
    words = sorted(list(set(words)))
    classes = sorted(list(set(classes)))
    
    print(f"Processed {len(words)} unique words, {len(classes)} classes, {len(documents)} patterns")
    
    return words, classes, documents

def create_training_data(words, classes, documents):
    """Create training data in the format required by neural network"""
    print("Creating training data...")
    
    training = []
    output_empty = [0] * len(classes)
    
    # Create training set
    for doc in documents:
        bag = []
        word_patterns = doc[0]
        word_patterns = [lemmatizer.lemmatize(word.lower()) for word in word_patterns]
        
        # Create bag of words
        for word in words:
            bag.append(1) if word in word_patterns else bag.append(0)
        
        # Create output row
        output_row = list(output_empty)
        output_row[classes.index(doc[1])] = 1
        training.append([bag, output_row])
    
    # Shuffle and convert to numpy array
    random.shuffle(training)
    training = np.array(training, dtype=object)
    
    # Split into X and Y
    train_x = list(training[:, 0])
    train_y = list(training[:, 1])
    
    print(f"Training data ready: {len(train_x)} samples with {len(train_x[0])} features")
    
    return train_x, train_y

# Process data
words, classes, documents = preprocess_data(intents)
train_x, train_y = create_training_data(words, classes, documents)

# =============================================================================
# NEURAL NETWORK MODEL
# =============================================================================

def create_model(input_shape, output_shape):
    """Create a neural network model for intent classification"""
    print("Building neural network...")
    
    model = Sequential()
    
    # Input layer with 128 neurons
    model.add(Dense(128, input_shape=(input_shape,), activation='relu'))
    model.add(Dropout(0.5))
    
    # Hidden layer with 64 neurons
    model.add(Dense(64, activation='relu'))
    model.add(Dropout(0.5))
    
    # Output layer
    model.add(Dense(output_shape, activation='softmax'))
    
    # Compile model (Fixed: Use legacy SGD)
    sgd = SGD(learning_rate=0.01, momentum=0.9, nesterov=True)
    model.compile(loss='categorical_crossentropy', optimizer=sgd, metrics=['accuracy'])
    
    print("Neural network created!")
    return model

def train_model(model, train_x, train_y, epochs=200):
    """Train the neural network model"""
    print(f"Training model for {epochs} epochs...")
    
    # Convert to numpy arrays
    train_x_array = np.array(train_x)
    train_y_array = np.array(train_y)
    
    # Train the model
    history = model.fit(
        train_x_array, 
        train_y_array, 
        epochs=epochs, 
        batch_size=5, 
        verbose=1
    )
    
    print(f"Training completed! Final accuracy: {history.history['accuracy'][-1]:.4f}")
    return history

# Create and train model
model = create_model(len(train_x[0]), len(train_y[0]))
history = train_model(model, train_x, train_y, epochs=200)

# Save model and data
print("Saving model and data...")
model.save('chatbot_model.h5')
pickle.dump(words, open('words.pkl', 'wb'))
pickle.dump(classes, open('classes.pkl', 'wb'))
print("Model and data saved!")

# =============================================================================
# CHATBOT FUNCTIONS
# =============================================================================

def clean_up_sentence(sentence):
    """Tokenize the pattern - split words into array"""
    sentence_words = nltk.word_tokenize(sentence)
    sentence_words = [lemmatizer.lemmatize(word.lower()) for word in sentence_words]
    return sentence_words

def bag_of_words(sentence, words, show_details=False):
    """Return bag of words array: 0 or 1 for each word in the bag that exists in the sentence"""
    sentence_words = clean_up_sentence(sentence)
    bag = [0] * len(words)
    
    for s in sentence_words:
        for i, word in enumerate(words):
            if word == s:
                bag[i] = 1
                if show_details:
                    print(f"Found in bag: {word}")
    
    return np.array(bag)

def predict_class(sentence, model, words, classes, show_details=False):
    """Filter out predictions below a threshold"""
    # Generate probabilities from the model
    p = bag_of_words(sentence, words, show_details)
    res = model.predict(np.array([p]), verbose=0)[0]
    
    ERROR_THRESHOLD = 0.25
    results = [[i, r] for i, r in enumerate(res) if r > ERROR_THRESHOLD]
    
    # Sort by strength of probability
    results.sort(key=lambda x: x[1], reverse=True)
    return_list = []
    
    for r in results:
        return_list.append({"intent": classes[r[0]], "probability": str(r[1])})
    
    return return_list

def get_response(ints, intents_json):
    """Get a response from the list of intents"""
    if len(ints) == 0:
        return "I'm sorry, I didn't understand that. Could you please rephrase?"
    
    tag = ints[0]['intent']
    list_of_intents = intents_json['intents']
    
    for i in list_of_intents:
        if i['tag'] == tag:
            result = random.choice(i['responses'])
            return result
    
    return "I'm not sure how to respond to that."

def chatbot_response(msg):
    """Generate a response from the chatbot"""
    ints = predict_class(msg, model, words, classes)
    res = get_response(ints, intents)
    return res

print("Chatbot functions ready!")

# =============================================================================
# TERMINAL INTERFACE
# =============================================================================

def terminal_chat():
    """Interactive terminal chat interface"""
    print("\n" + "=" * 60)
    print("TERMINAL CHATBOT READY!")
    print("=" * 60)
    print("Start chatting! (Type 'quit', 'exit', or 'bye' to stop)")
    print("-" * 60)
    
    conversation_count = 0
    
    while True:
        # Get user input
        user_input = input("\nYou: ").strip()
        
        # Check for exit commands
        if user_input.lower() in ['quit', 'exit', 'bye', 'stop', 'end']:
            farewell = chatbot_response("goodbye")
            print(f"Bot: {farewell}")
            print(f"\nChat ended after {conversation_count} exchanges. Thanks for chatting!")
            break
        
        # Check for empty input
        if not user_input:
            print("Bot: Please say something! I'm here to help.")
            continue
        
        # Get chatbot response
        try:
            response = chatbot_response(user_input)
            print(f"Bot: {response}")
            conversation_count += 1
        except Exception as e:
            print(f"Bot: Sorry, I encountered an error: {e}")

# =============================================================================
# GUI INTERFACE
# =============================================================================

class ChatbotGUI:
    def __init__(self, root):
        self.root = root
        self.root.title("AI Chatbot - VS Code Edition")
        self.root.geometry("600x700")
        self.root.resizable(True, True)
        
        # Configure colors
        self.bg_color = "#f0f8ff"
        self.chat_bg = "white"
        self.user_color = "#007bff"
        self.bot_color = "#28a745"
        
        self.root.configure(bg=self.bg_color)
        self.setup_gui()
        
    def setup_gui(self):
        # Title
        title_label = tk.Label(
            self.root, 
            text="AI Chatbot Assistant", 
            font=("Arial", 18, "bold"), 
            bg=self.bg_color, 
            fg="#333333"
        )
        title_label.pack(pady=15)
        
        # Subtitle
        subtitle_label = tk.Label(
            self.root, 
            text="Powered by Neural Networks & Natural Language Processing", 
            font=("Arial", 10), 
            bg=self.bg_color, 
            fg="#666666"
        )
        subtitle_label.pack(pady=(0, 10))
        
        # Chat display area
        self.chat_display = scrolledtext.ScrolledText(
            self.root, 
            width=70, 
            height=25,
            font=("Arial", 11),
            bg=self.chat_bg,
            state=tk.DISABLED,
            wrap=tk.WORD
        )
        self.chat_display.pack(pady=10, padx=15, fill=tk.BOTH, expand=True)
        
        # Input frame
        input_frame = tk.Frame(self.root, bg=self.bg_color)
        input_frame.pack(pady=15, padx=15, fill=tk.X)
        
        # Message input
        self.message_entry = tk.Entry(
            input_frame, 
            font=("Arial", 12), 
            width=50
        )
        self.message_entry.pack(side=tk.LEFT, fill=tk.X, expand=True, padx=(0, 10))
        self.message_entry.bind("<Return>", self.send_message)
        
        # Send button
        send_button = tk.Button(
            input_frame, 
            text="Send", 
            font=("Arial", 12, "bold"),
            bg="#007bff", 
            fg="white",
            width=10,
            command=self.send_message
        )
        send_button.pack(side=tk.RIGHT)
        
        # Status bar
        self.status_label = tk.Label(
            self.root, 
            text="Chatbot ready! Type a message...", 
            font=("Arial", 10), 
            bg=self.bg_color, 
            fg="#666666"
        )
        self.status_label.pack(pady=5)
        
        # Welcome message
        self.add_message("Bot", "Hello! I'm your AI assistant powered by neural networks. How can I help you today?", self.bot_color)
        
        # Focus on input
        self.message_entry.focus()
        
    def add_message(self, sender, message, color):
        self.chat_display.config(state=tk.NORMAL)
        
        # Add timestamp
        timestamp = datetime.now().strftime("%H:%M")
        
        # Add message with formatting
        self.chat_display.insert(tk.END, f"[{timestamp}] {sender}: ", (sender.lower(),))
        self.chat_display.insert(tk.END, f"{message}\n\n")
        
        # Configure tags for colors
        self.chat_display.tag_configure(sender.lower(), foreground=color, font=("Arial", 11, "bold"))
        
        self.chat_display.config(state=tk.DISABLED)
        self.chat_display.see(tk.END)
        
    def send_message(self, event=None):
        message = self.message_entry.get().strip()
        
        if not message:
            return
            
        # Add user message
        self.add_message("You", message, self.user_color)
        
        # Clear input
        self.message_entry.delete(0, tk.END)
        
        # Update status
        self.status_label.config(text="Thinking...")
        self.root.update()
        
        # Get bot response
        try:
            response = chatbot_response(message)
            self.add_message("Bot", response, self.bot_color)
            self.status_label.config(text="Message sent!")
        except Exception as e:
            self.add_message("Bot", "Sorry, I encountered an error. Please try again.", "red")
            self.status_label.config(text=f"Error: {str(e)}")
        
        # Reset status after 2 seconds
        self.root.after(2000, lambda: self.status_label.config(text="Type a message..."))

def launch_gui():
    """Launch the GUI chatbot"""
    try:
        root = tk.Tk()
        app = ChatbotGUI(root)
        root.mainloop()
    except Exception as e:
        print(f"GUI Error: {e}")
        print("Try using terminal mode instead")

# =============================================================================
# PERFORMANCE TESTING
# =============================================================================

def test_chatbot():
    """Test the chatbot with sample messages"""
    print("\nTESTING CHATBOT PERFORMANCE...")
    print("=" * 50)
    
    test_messages = [
        "Hello!",
        "What's your name?", 
        "Can you help me?",
        "Tell me a joke",
        "What can you do?",
        "How's the weather?",
        "What time is it?",
        "Thank you very much",
        "You're awesome!",
        "Goodbye"
    ]
    
    start_time = datetime.now()
    
    for i, message in enumerate(test_messages, 1):
        response = chatbot_response(message)
        print(f"{i:2d}. User: {message}")
        print(f"    Bot:  {response}")
        print("-" * 40)
    
    end_time = datetime.now()
    total_time = end_time - start_time
    avg_time = total_time.total_seconds() / len(test_messages)
    
    print(f"PERFORMANCE RESULTS:")
    print(f"   Total Time: {total_time}")
    print(f"   Average Response Time: {avg_time:.3f} seconds")
    print(f"   Messages Processed: {len(test_messages)}")

# =============================================================================
# PLOTTING FUNCTIONS
# =============================================================================

def plot_training_history():
    """Plot training accuracy and loss"""
    if not PLOTTING_AVAILABLE:
        print("Matplotlib not available for plotting")
        return
    
    try:
        plt.figure(figsize=(12, 4))
        
        # Plot accuracy
        plt.subplot(1, 2, 1)
        plt.plot(history.history['accuracy'], 'b-', linewidth=2)
        plt.title('Model Accuracy', fontsize=14, fontweight='bold')
        plt.ylabel('Accuracy')
        plt.xlabel('Epoch')
        plt.grid(True, alpha=0.3)
        plt.ylim(0, 1)
        
        # Plot loss
        plt.subplot(1, 2, 2)
        plt.plot(history.history['loss'], 'r-', linewidth=2)
        plt.title('Model Loss', fontsize=14, fontweight='bold')
        plt.ylabel('Loss')
        plt.xlabel('Epoch')
        plt.grid(True, alpha=0.3)
        
        plt.tight_layout()
        plt.show()
        
        print("Training plots displayed!")
        
    except Exception as e:
        print(f"Plotting error: {e}")

# =============================================================================
# MAIN INTERFACE SELECTION
# =============================================================================

def main_menu():
    """Main menu for selecting chatbot interface"""
    print("\n" + "=" * 50)
    print("      CHATBOT PROJECT COMPLETED!")
    print("=" * 50)
    
    print(f"""
AI CHATBOT READY FOR USE!
{'='*50}

Model Statistics:
   • Training Accuracy: {history.history['accuracy'][-1]:.1%}
   • Training Loss: {history.history['loss'][-1]:.4f}
   • Vocabulary Size: {len(words)} words
   • Intent Categories: {len(classes)}
   • Training Samples: {len(train_x)}

Available Interfaces:
   1. Terminal Chat (Text-based)
   2. GUI Chat (Desktop Window)
   3. Test Performance
   4. Show Training Plots
   5. Exit

{'='*50}
""")
    
    while True:
        try:
            choice = input("Select an option (1-5): ").strip()
            
            if choice == '1':
                terminal_chat()
            elif choice == '2':
                print("Launching GUI chatbot...")
                launch_gui()
            elif choice == '3':
                test_chatbot()
            elif choice == '4':
                plot_training_history()
            elif choice == '5':
                print("Goodbye! Thanks for using the AI Chatbot!")
                break
            else:
                print("Invalid choice. Please select 1-5.")
                
            if choice in ['1', '2', '3', '4']:
                input("\nPress Enter to return to main menu...")
                main_menu()
                break
                
        except KeyboardInterrupt:
            print("\n\nGoodbye! Thanks for using the AI Chatbot!")
            break
        except Exception as e:
            print(f"Error: {e}")

# =============================================================================
# QUICK FUNCTIONS FOR TESTING
# =============================================================================

def quick_chat(message):
    """Quick chat function for testing"""
    return chatbot_response(message)

def batch_test(messages):
    """Test multiple messages at once"""
    results = []
    for msg in messages:
        response = chatbot_response(msg)
        results.append({"input": msg, "output": response})
    return results

# Create some sample variables
sample_test_messages = [
    "Hello!",
    "What's your name?",
    "Tell me a joke",
    "Thank you",
    "Goodbye"
]

sample_responses = batch_test(sample_test_messages)

print("\n" + "=" * 50)
print("      ALL SYSTEMS READY!")
print("=" * 50)

print(f"""
CONGRATULATIONS! Your AI Chatbot is ready!

Quick Usage:
   • Test single message: quick_chat("Hello!")
   • Run full interface: main_menu()
   • Terminal chat: terminal_chat()
   • GUI chat: launch_gui()

Example Usage:
   >>> quick_chat("Hello, how are you?")
   >>> terminal_chat()
   >>> main_menu()
""")

# =============================================================================
# AUTO-START SECTION
# =============================================================================

if __name__ == "__main__":
    # If running as main script, show menu
    try:
        main_menu()
    except KeyboardInterrupt:
        print("\nGoodbye!")
else:
    # If imported, just show ready message
    print("\nChatbot loaded! Use quick_chat('message') or main_menu()")

# =============================================================================
# END OF CHATBOT PROJECT
# Ready to use!
# =============================================================================