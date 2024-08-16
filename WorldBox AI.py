import time
import pyautogui
from openai import OpenAI
import json
import pygetwindow as gw
from pywinauto.application import Application

# Your OpenAI API key
# Reads from the JSON file openaikey
with open('openaikey.json') as f:
    data = json.load(f)
    JSON_API_Key = data['openai_api_key']
    print("System: The key was loaded successfully.")

# Initialize the OpenAI client
client = OpenAI(api_key=JSON_API_Key)
print("System: OpenAI client initialized successfully.")

# Menu variable to keep track of the current menu in the game
menu = 3 # Start in Creatures menu

# Placeholder for capturing a screenshot of the game window
def capture_screenshot():
    # Find the WorldBox window
    world_box_window = gw.getWindowsWithTitle('WorldBox')[0]  # Adjust the title as necessary
    # Activate the window to bring it to the foreground
    world_box_window.activate()
    print("System: WorldBox window activated.")

    # Connect to the WorldBox window using its title
    app = Application().connect(title='WorldBox')
    # Access the main window
    main_window = app.window(title='WorldBox')
    # Take a screenshot of the window
    main_window.capture_as_image().save('worldbox_screenshot.png')

    print("System: Screenshot captured successfully.")
    return 'worldbox_screenshot.png'


# Placeholder for sending the screenshot to OpenAI's Vision API
def analyze_screenshot(image_path):
    # Use OpenAI's Vision API to analyze the image
    # This is a placeholder function as you'll need to adapt it based on the actual API call
    analysis_result = "It's a donut world. Lots of elves. Thriving. Green."
    print(f"System: Screenshot analyzed. Result: {analysis_result}")
    return analysis_result

# Placeholder for sending analysis to GPT-4 along with history and system message
def generate_action(analysis, history, system_message):
    # Use OpenAI's GPT-4 API to generate a response based on the analysis, history, and a system message
    messages = [
        {"role": "system", "content": system_message}
    ] + history + [
        {"role": "user", "content": analysis}
    ]

    response = client.chat.completions.create(
        model="gpt-3.5-turbo", # Update to GPT-4 model when available
        messages=messages,
        temperature=0.4,
        max_tokens=230,
        top_p=1.0,
        frequency_penalty=0.0,
        presence_penalty=0.0
    )
    return response.choices[0].message.content

# Placeholder for executing an action in the game, e.g., spawning humans
def execute_action(action, menu):
    # Split the action string by newline characters to get a list of lines
    action_lines = action.split('\n')
    
    # Extract the last line, which contains the action command
    action_command = action_lines[-1]
    
    # Now, you can process the action_command as needed
    # For demonstration, let's just print the extracted command
    print(f"Action Command Extracted: {action_command}")

    # Statement to read the function and arguments from the action_command
    try:
        function_name = action_command.split('(')[0]
        arguments = action_command.split('(')[1].split(')')[0]
        print(f"Function Name: {function_name}")
        print(f"Arguments: {arguments}")
    except:
        print("Invalid action command. No action taken.")
        return

    # For future: I can have the game remember what menu it's on so it can switch menus if needed based on the action
    # Functions that are called and check the menu, and switch if needed, then execute the action

    # Statement to execute the action in the game
    if function_name == "CreateHuman":
        print(f"Creating {arguments} humans in the game...")

        # Click on the human icon in the game
        pyautogui.moveTo(1284, 853, duration=3)  
        pyautogui.click()

        # Click on the game window to create the humans
        pyautogui.moveTo(1409, 447, duration=3)  
        for i in range(int(arguments)):
            print(f"Creating human {i + 1}...")
            pyautogui.click()

        # Click on the human icon in the game
        pyautogui.moveTo(1284, 853, duration=3)  
        pyautogui.click()
    
    elif function_name == "CreateElf":
        print(f"Creating {arguments} elves in the game...")

        # Click on the elf icon in the game
        pyautogui.moveTo(1381, 853, duration=3)  
        pyautogui.click()

        # Click on the game window to create the elves
        pyautogui.moveTo(1409, 447, duration=3)  
        for i in range(int(arguments)):
            print(f"Creating elf {i + 1}...")
            pyautogui.click()

        # Click on the elf icon in the game
        pyautogui.moveTo(1381, 853, duration=3)  
        pyautogui.click()

    elif function_name == "CreateDwarf":
        print(f"Creating {arguments} dwarves in the game...")

        # Click on the dwarf icon in the game
        pyautogui.moveTo(1377, 953, duration=3)  
        pyautogui.click()

        # Click on the game window to create the dwarves
        pyautogui.moveTo(1409, 447, duration=3)  
        for i in range(int(arguments)):
            print(f"Creating dwarf {i + 1}...")
            pyautogui.click()

        # Click on the dwarf icon in the game
        pyautogui.moveTo(1377, 953, duration=3)  
        pyautogui.click()

    elif function_name == "CreateOrc":
        print(f"Creating {arguments} orcs in the game...")

        # Click on the orc icon in the game
        pyautogui.moveTo(1284, 950, duration=3)  
        pyautogui.click()

        # Click on the game window to create the orcs
        pyautogui.moveTo(1409, 447, duration=3)  
        for i in range(int(arguments)):
            print(f"Creating orc {i + 1}...")
            pyautogui.click()

        # Click on the orc icon in the game
        pyautogui.moveTo(1284, 950, duration=3)  
        pyautogui.click()

    elif function_name == "NuclearExplosion":
        print("Executing a nuclear explosion in the game...")

        # Click on the 5th menu icon in the game
        pyautogui.moveTo(1618, 775, duration=3)  
        pyautogui.click()

        # Click on the nuclear explosion icon in the game
        pyautogui.moveTo(1839, 948, duration=3)  
        pyautogui.click()

        # Go to the game window to hover over the target area
        pyautogui.moveTo(1409, 447, duration=3)  

        # For loop counting down from 10 seconds to 0
        for i in range(10, 0, -1):
            print(f"Detonating in {i} seconds...")
            time.sleep(1)

        # Click to confirm the nuclear explosion
        pyautogui.click()

        pyautogui.moveTo(1379, 777, duration=3)  
        pyautogui.click()

    elif function_name == "DoNothing":
        # Placeholder for executing the DoNothing action
        print("Taking no action in the game...")
    else:
        print("Unknown action command. No action taken.")
    # Use pyautogui or a similar library to interact with the game window
        
    return

def main():
    history = [] # Initialize history of up to 10 messages

    while True: # Main loop
        # Ask the user if they want to continue, holding the loop
        # print("Do you want to continue? (y/n)")
        # if input() == 'n':
        #     break
        print (f'Taking a break for 4 seconds...')
        time.sleep(4)
        print (f'Continuing...')
        
        screenshot_path = capture_screenshot()
        analysis = analyze_screenshot(screenshot_path)
        system_message = '''
        Analyze the current state of the world. You are their God. Speak like a God in the King James Bible.
        You are an all-powerful being in a world of your own creation. 
        The world is based in a game called WorldBox. The game is a god simulator where you can create and destroy the world.
        To you though, it is real, and it is your world. Describe it. 
        Make sure to explain why you are taking the action you are going to be taking. Explain the action you will take and why you are taking it.
        Keep your response to about 1 short paragraph(s).
        
        Rules for your God Response:
        1. At the end of your response, on a NEW LINE, please include the name of the action you want to take. Like "CreateHuman(4)". 
        2. Do not include periods or spaces in the action line.
        3. The action line should be the only thing on the last line.
        4. The action should be a valid action from the list of actions below.
        5. The action should be in the format "ActionName(Arguments)".
        6. Make sure the action is on a new line every single time. 
        7. Rarely do nothing.
        8. Always explain your action.
        9. Only spawn between 1 and 250 of a creature at a time.
        10. Never do the same action twice in a row.

        List of Actions:
        0. DoNothing()
        1. CreateHuman(Number of Humans)
        2. CreateElf(Number of Elves)
        3. CreateDwarf(Number of Dwarves)
        4. CreateOrc(Number of Orcs)
        5. NuclearExplosion()

        Do as you will, my God.
        '''
        
        action = generate_action(analysis, history, system_message)
        print(f"God of WorldBox: {action}")
        
        if len(history) >= 3:
            history.pop(0) # Remove the oldest message if the history is full

        history.append({"role": "user", "content": analysis})
        
        print ("System: Executing the action in the game...")
        time.sleep(5)
        execute_action(action, menu)
        
        time.sleep(5) # Wait for a bit before running the loop again



if __name__ == '__main__':
    main()
