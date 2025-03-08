import sys
from PyQt5.QtWidgets import QApplication
from ui import RubikAIInterface
from complete_advanced_rubiks_bot import AdvancedRubiksNLUBot

def main():
    # Initialize the bot with knowledge
    bot = AdvancedRubiksNLUBot(knowledge_file='knowledge.json')
    
    # Create Qt application
    app = QApplication(sys.argv)
    
    # Create and show the main window
    window = RubikAIInterface(bot)
    window.show()
    
    # Start the event loop
    sys.exit(app.exec_())

if __name__ == "__main__":
    main()