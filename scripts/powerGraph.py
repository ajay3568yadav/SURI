import re
import os
import sys
import pickle
sys.path.append("classes")
from pin import Pin
from delayobject import DelayObject
from node import Node

# extracts string between quotes
def extract_string_between_quotes(string):
    pattern = r'"([^"]*)"'
    match = re.search(pattern, string)
    if match:
        return match.group(1)
    else:
        return None

#return the key from a line of certain format
def extract_key(string):
    li = string.strip().split()
    if len(li)>1:
        x=li[1].replace("(","")
        x=x.replace(")","")
        return x
    return string.strip().split()[0]


#extracts the largest number from a string of liberty "value" attribute format
def extract_largest_number(string):
    numbers = re.findall(r'-?\d+(?:\.\d+)?', string)
    if numbers:
        largest_number = max(map(float, numbers))
        return largest_number
    else:
        return None


def cell_check(line:str):
    return line.find("cell (") >= 0 or line.find("internal_power") >= 0 or line.find(
                        "related_pin") >= 0 or line.find("related_pg_pin") >= 0

# Return the cell to max power dictionary for the input file
def process_file(filename):
    """
    cell_pin_dictionary has mapping of cell to the maximum power 
    """
    cell_pin_dictionary = {}
    with open(filename, "r") as f:
        # Filtering the information from the liberty file
        with open("cellmap.txt", "w") as w:
            line = f.readline()
            
            while line:
                """
                if the line contains cell name, or internal_power, related_pin or related_pg_pin
                write the line to the file.
                """
                if cell_check(line):
                    w.writelines(line)
                else:
                    if line.find("rise_power (") >= 0:
                        # Default max value set to minimum possible for comparison and alteration
                        maxnum = -1000
                        # "}" marks the end of the rise power table
                        while line.find("}") < 0:
                            if line.find("\\") >= 0 and line.find("values") < 0:
                                maxnum = extract_largest_number(line)
                            line = f.readline()
                        # Writing into the max risevalue field   
                        w.writelines("max risevalue: " + str(maxnum) + "\n")
                    elif line.find("fall_power (") >= 0:
                        # "}" marks the end of the fall power table
                        while line.find("}") < 0:
                            if line.find("\\") >= 0 and line.find("values") < 0:
                                maxnum = extract_largest_number(line)
                            line = f.readline()
                        # Writing into the max fallvalue field
                        w.writelines("max fallvalue: " + str(maxnum) + "\n}\n")
                        
                line = f.readline()
        w.close()
        
    # Condensing the information in the cellmap file
    with open("cellmap.txt", "r") as f:
        with open("cellmap2.txt", "w") as w:
            line = " "
            while line:
                line = f.readline()
                if line.find("cell (") >= 0:
                    w.writelines(line)
                    while line.find("internal_power") < 0:
                        line = f.readline()
                w.writelines(line)
        w.close()
            
    # Extracting values from cellmap2 fields to the power dictionary
    # Open the file "cellmap2.txt" in read mode
    with open("cellmap2.txt", "r") as f:
        line = "xx"  # Initialize the line variable
        key = ""     # Initialize the key variable
    
        
        while line:
            
            # Check if the line contains the string "cell ("
            if line.find("cell (") >= 0:
                
                # Extract the key using the extract_key function
                key = extract_key(line)
                
                # Initialize a list to store power-related information
                powerlist = []  
                line = f.readline()  
            
                # Iterate until the next cell definition or the end of the file
                while line.find("cell (") < 0 and line:
                    
                    # Check if the line contains the string "internal_power"
                    if line.find("internal_power") >= 0:
                        
                        pin = " "  # Initialize pin variable
                        rv = 0     # Initialize rise value variable
                        fv = 0     # Initialize fall value variable
                        pg = ""    # Initialize power group variable
                    
                        while line.find("}") < 0 and line:
                            
                            line = f.readline()  # Read the next line
                            
                            # Extract information based on keywords in the line
                            if line.find("related_pin") >= 0:
                                pin = extract_string_between_quotes(line)
                            if line.find("VDD") >= 0:
                                pg = "VDD"
                            if line.find("VSS") >= 0:
                                pg = "VSS"
                            if line.find("risevalue") >= 0:
                                rv = float(line.strip().split()[-1])
                            if line.find("fallvalue") >= 0:
                                fv = float(line.strip().split()[-1])
                    
                        # Append the Pin object to the powerlist
                        powerlist.append(Pin(pg, rv, fv, pin))
                
                    line = f.readline() 
            
                # Add the powerlist to the cell_pin_dictionary with the key
                cell_pin_dictionary[key] = powerlist
            else:
                line = f.readline()  
    
    # Return the generated cell_pin_dictionary
    return cell_pin_dictionary
 
#return the absolute paths of the file in the lib folder containing liberty files
def get_absolute_paths(folder_path):
    
    absolute_paths = []

    for root, dirs, files in os.walk(folder_path):
        
        for file in files:
            
            absolute_path = os.path.abspath(os.path.join(root, file))
            
            absolute_paths.append(absolute_path)

    return absolute_paths

#return the list of the absolute paths of the files in the lib folder containing liberty files
def get_liberty_files():
    
    external_folder_path = "../lib"
    
    path_list = []

    try:
        absolute_paths = get_absolute_paths(external_folder_path)

        for path in absolute_paths:
            
            # Case-insensitive check for "asap" in the path
            if "asap" in path.lower(): 
                
                path_list.append(path)

    except Exception as e:
        pass
    
    return path_list

liberty_list = get_liberty_files()

"""
cell_pin_dictionary maps cell to list of pins associated with the cell in the lib file

<cell_name> : [<pin_object>,...]

"""
cell_pin_dictionary = {}

for file in liberty_list:
    cell_pin_dictionary.update(process_file(file))

# Function to extract text between parentheses
def extract_text_between_parentheses(input_text):
    pattern = r'\((.*?)\)'
    matches = re.findall(pattern, input_text)
    if matches:
        return matches[0]  # Return the first match as a string
    else:
        return None

# Function to extract the max capacitance value from input text
def extract_max_capacitance(input_text):
    pattern = r'max_capacitance : (\d+\.\d+);'
    match = re.search(pattern, input_text)
    if match:
        return float(match.group(1))
    else:
        return None
    
# Function to create a dictionary mapping cells to their respective max capacitance values
def get_max_cap_dict(filelist):
    maxcapdict = {}
    for x in filelist:
        with open(x, "r") as f:
            line = f.readline()
            cell = ""
            while line:
                # Extract cell name between parentheses
                if " cell (" in line:
                    cell = extract_text_between_parentheses(line)
                
                # Extract max capacitance value from the line
                if "max_capacitance" in line:
                    maxcap = extract_max_capacitance(line)
                    # Add the mapping to the dictionary if both cell and maxcap values are present
                    if cell is not None and maxcap is not None:
                        maxcapdict[cell] = maxcap
                line = f.readline()
    return maxcapdict

# Generate a dictionary mapping cells to their max capacitance values from a list of files
max_cap_dict = get_max_cap_dict(liberty_list)


"""
max_power_dictionary maps cell to max power associated with the cell in the lib file

<cell_name> : max_power

"""
max_power_dictionary={}

for cell in cell_pin_dictionary:
    
    #setting max_power to minimum supposed value
    max_power = -1000
    
    for pin in cell_pin_dictionary[cell]:
        
        #considering only VDD pins
        if pin.pg == "VDD":
            
            #comparing fall and rise values with the current max power
            max_power = max(pin.rv,pin.fv,max_power)
            
    # mapping cell to its maximum power value        
    max_power_dictionary[cell] = max_power

#return the abbrevation for the asap cell, for mapping with appropriate gen dummy cell
def get_characters_until_x(string):
    i = string.find("x")
    return string[:i]


def get_cell_abv_dict(key_list:list):
    cell_abv_dictionary = {}
    for key in key_list:
        abv = get_characters_until_x(key)
        cell_abv_dictionary[abv] = key
    return cell_abv_dictionary


# asap cell abv to cell for matching dummy cells to corresponding asap7 cell
cell_abv_dictionary = get_cell_abv_dict(list(max_power_dictionary.keys()))


# all unique gen cells to their 
gen_cell_dictionary={} 
try:
    with open("../files/ExampleRocketSystem_GEN.gv", "r") as filereader:
        line = filereader.readline()
        while line:
            if line.strip()[:4]=="GEN_":
                gen_cell =line.strip().split()[0]
                gen_abv = re.search(r'(?<=_)[^_]+', gen_cell).group()
                gen_cell_dictionary[gen_cell] = gen_abv
            line = filereader.readline()
except FileNotFoundError:
    pass
except IOError:
    pass


#contain gen cell to asap cell maps for replacing the dummy cells in the ExampleRocketSystem_GEN file
map_dictionary={} 

for gen_cell in list(gen_cell_dictionary.keys()):
    for cell in list(cell_abv_dictionary.values()):
        if gen_cell.find("MAJORITY")>=0:
            gen_cell_dictionary[gen_cell]=gen_cell_dictionary[gen_cell].replace("MAJORITY","")
        if cell[:len(gen_cell_dictionary[gen_cell])] == gen_cell_dictionary[gen_cell]: 
            map_dictionary[gen_cell]= cell
            

# mapping unmatched cells manually
map_dictionary["GEN_DFCLR_D1"]="AND3x1_ASAP7_75t_L" #4 ports
map_dictionary["GEN_XNOR3_D1"]="AND3x1_ASAP7_75t_L" #4 ports
map_dictionary["GEN_OA32_D1"] = "A2O1A1O1Ixp25_ASAP7_75t_L"
map_dictionary["GEN_XOR3_D1"] = "OA21x2_ASAP7_75t_L"# 6 ports i guess
map_dictionary["GEN_MUX4_D1"]="AO222x2_ASAP7_75t_L" #7 ports
map_dictionary["GEN_MUX2_D2"]="AND3x1_ASAP7_75t_L"#4 ports
map_dictionary["GEN_DFCLR_D2"]="AND3x1_ASAP7_75t_L"# 4 ports
map_dictionary["GEN_LATCH_D1"]="AND2x2_ASAP7_75t_L"
map_dictionary["GEN_RAMS_SP_WENABLE8_4096x32"]="A2O1A1O1Ixp25_ASAP7_75t_L"
map_dictionary["GEN_RAMS_SP_WENABLE21_64x21"]="AO222x2_ASAP7_75t_L" #7ports
map_dictionary["GEN_RAMS_SP_WENABLE32_1024x32"] = "AO222x2_ASAP7_75t_L" #7ports
map_dictionary["GEN_XOR3_D1"]="AND3x1_ASAP7_75t_L"#4 ports
map_dictionary["GEN_NAND2_D2"]="NAND2xp5_ASAP7_75t_L"#3ports


#find and replace the dummy cell with the appropriate asap7 technology
def replace_with_asap(line:str,map_dictionary:dict):
    dummy_line=line.strip()
    if dummy_line[:4]=="GEN_":
        dummy_line_list=dummy_line.split()
        if dummy_line_list[0] in list(map_dictionary.keys()):
            asapcell=map_dictionary[dummy_line_list[0]]
            line=line.replace(dummy_line_list[0],asapcell)
    return line


#running this function creates the netlist with the replaced dummy cells
def get_netlist():
    with open("../files/ExampleRocketSystem_GEN.txt","w") as writer:
        with open("../files/ExampleRocketSystem_GEN.gv","r") as reader:
            lines=reader.readline()
            while lines:
                
                #write lines searching and replacing the GEN cells
                writer.writelines(replace_with_asap(lines,map_dictionary))
                lines=reader.readline()
            writer.close()

#running get_netlist to generate the dummy file in the respective folder
get_netlist()

with open("../files/ExampleRocketSystem_GEN.gv","r") as f:
    line=f.readline()
    with open("../test/test_gen_file.txt","w") as w:
        while line:
            if line.strip()[0:4] == "GEN_":
                if line.find(";") > 0:
                    w.write(replace_with_asap(line,map_dictionary))
                else:
                    l=line.rstrip()
                    while line.find(";") < 0:
                        line=f.readline()
                        l=l+line.strip()
                    w.write(replace_with_asap(l,map_dictionary)+"\n")
            else:
                w.write(replace_with_asap(line,map_dictionary))
            line=f.readline()
            
            
def get_range(input_string:str):
    # Input string
    if input_string == "":
        return
    # Define a regular expression pattern to match both numbers within square brackets
    pattern = r'\[(\d+):(\d+)\]'
    # Use re.search to find the match
    match = re.search(pattern, input_string)
    # If there is a match, extract both numbers from the capturing groups
    if match:
        number1 = int(match.group(1))
        number2 = int(match.group(2))
        return [number1, number2]
    else:
        return
    

# Load the objectdict dictionary from parsingfile.ipynb
with open('../newPickle/object_dictionary.pickle', 'rb') as f:
    object_dictionary = pickle.load(f)
    
#retuns wires connected to the given net
def get_connected_wires(input_string):
    pattern = r'\.([a-zA-Z0-9_]+)\(([^)]+)\)'
    matches = re.findall(pattern, input_string)
    li=[]
    for match in matches:
        li.append(match[1])
    return li

# to remove the [] associated with wires with range/ or busses
def remove_square_brackets(input_string):
    pattern = r'\[\d{1,3}\]'
    
    match = re.search(pattern, input_string)
    
    if match:
        modified_string = re.sub(pattern, '', input_string)
        return modified_string
    else:
        return input_string

#returns the list of delay object corresponding to a given output wire(name of the output wire)
def get_delay_object(name):
    
    dummy=name
    li=[]
    for radix in object_dictionary:
        
        object_list=object_dictionary[radix]
        
        for i in range(len(object_list)):
            
            #accounting for busses with different port numbers
            if object_list[i].name == remove_square_brackets(dummy):
                
                li.append(object_list[i])
    return li 
    


"""
node_dictionary contains the mapping of node output wire to the Node object

"""
    
node_dictionary = {}

#using the intermediate file for easier operation and faster processing
with open("../test/test_gen_file.txt","r") as reader:
    
    line=reader.readline()
    
    while line:
        
        if line.find("ASAP7") > 0:
            current_line_split=line.strip().split()
            
            #name of the net
            name = current_line_split[1]
            
            #cell type
            cell = current_line_split[0]
            
            #peak cell power
            peak = max_power_dictionary[cell]
            
            #list of wires connected to the net; both input and output
            inputlist = get_connected_wires(line)
            
            #output wire attached to the net
            output=inputlist[-1]
            
            #inputs for a given net
            inputs=inputlist[:-1]
            
            #maximum capacitance for the give net
            maxcap = max_cap_dict[cell]
            
            #unknown radix and toggle dict values
            radix,toggle="",{}
            
            #retriving the delay object which is a list
            delay_object = get_delay_object(output)
                
            for i in range(len(delay_object)):
                radix=delay_object[i].radix
                toggle=delay_object[i].toggle
                if output not in list(node_dictionary.keys()):
                    node_dictionary[output]=[]
                node_dictionary[output].append(Node(name, 
                                                    cell, 
                                                    output, 
                                                    inputs, 
                                                    radix, 
                                                    toggle, 
                                                    maxcap=maxcap, 
                                                    peak=peak, 
                                                    t1=0.27696088077503833, 
                                                    t2=0.45389178609342473))
        line=reader.readline()

"""
Info about the extra/ unusual node:

output = "debug_1_ICCADs_dmOuter_ICCADs_dmiXbar_ICCADs_n2"
name = "debug_1_ICCADs_dmOuter_ICCADs_dmiXbar_ICCADs_U5"
cell = "INVxp67_ASAP7_75t_SL"
inputs = ["debug_1_ICCADs_dmOuter_ICCADs_dmiXbar_ICCADs_n15"]
maxcap = max_cap_dict[cell]
radix = "X##"
peak = max_power_dictionary[cell]
node_dictionary["debug_1_ICCADs_dmOuter_ICCADs_dmiXbar_ICCADs_n2"].append(Node(name,cell,output,
                                            inputs,radix,toggle,maxcap,peak,t1=0.27696088077503833, 
                                                    t2=0.45389178609342473))
"""

node_dictionary["debug_1_ICCADs_dmOuter_ICCADs_dmiXbar_ICCADs_n2"].pop()


# Fixing missing inputs
def remove_square_brackets_content(input_string):
    # Use a regex pattern to match brackets and their contents at the end of the string
    pattern = r'\[[^\]]*\]$'
    
    # Search for the pattern in the input_string
    match = re.search(pattern, input_string)
    
    if match:
        # Replace the matched pattern with an empty string
        modified_string = re.sub(pattern, '', input_string)
        return modified_string
    else:
        return input_string
    
# List to store all wires that are inputted into a cell/gate 
all_input_wires = []

# Extract all input wires from the node_dictionary
for node in node_dictionary:
    all_input_wires = all_input_wires + node_dictionary[node][0].inputs

# Remove duplicates by converting the list to a set
all_input_wires_set = set(all_input_wires)

# List to store wires that are not output wires
missing_wires = []

# Identify missing wires by checking if they are not output wires of any net
for wire in all_input_wires_set:
    if wire not in node_dictionary:
        missing_wires.append(wire)

# Dictionary to map wire names to corresponding objects
objnamedict = {}

# Populate objnamedict using object_dictionary
for radix in object_dictionary:
    try:
        for obj in object_dictionary[radix]: 
            objnamedict[obj.name] = obj
    except KeyError as e:
        print(f"KeyError: {e.name}")

# Iterate through missing wires and add them to the node_dictionary with default values
for wire in missing_wires:
    # Skip certain wire types
    if wire[0] == "{" or wire == "1'b1" or wire == "1'b0":
        continue
    
    # Get the object associated with the wire
    obj = objnamedict[remove_square_brackets_content(wire)]
    
    # Add the missing wire to node_dictionary with default Node values
    node_dictionary[wire] = [Node("no_output: " + wire,
                                  "",
                                  wire,
                                  [],
                                  obj.radix,
                                  obj.toggle,
                                  t1=0.27696088077503833, 
                                  t2=0.45389178609342473)]
    

with open('../newPickle/node_dictionary.pkl', 'wb') as file:
    pickle.dump(node_dictionary, file)