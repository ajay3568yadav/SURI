import sys

import pickle

sys.path.append("classes")
from delayobject import DelayObject

file = "../files/ExampleRocketSystem_rando_sampleOutput.v"

f= open(file,"r")
current_line=f.readline().rstrip("\n")

while current_line[:4]!="$var":
    
    current_line=f.readline().rstrip("\n")
    
""" 

Objectdict structure is as follow:

object_dictionary: type: dict

<key> : Radix, <value>: delay object/ object of class DelayObject

"""

object_dictionary = {}

end_of_file = False
try:
    while not end_of_file:
        current_line_string_list = current_line.split()
        
        width = ""
        radix = ""
        name = ""
        range_ = ""
        
        """ 
        
        if there are more than 6 entries/strings in one line 
        then for the given line range exists 
        and needs to be considered 
        
        """
        if len(current_line_string_list) >= 6:
            
            # range exists is the 6th string in the line is not "$end"
            if current_line[5] != "$end":
                
                width = current_line_string_list[2]
                radix = current_line_string_list[3]
                name = current_line_string_list[4]
                range_ = current_line_string_list[5]  
                
            #create DelayObject 
            delay_object = DelayObject(width, radix, name, range_)
            
            # mapping the objects to their radix
            if delay_object.radix not in list(object_dictionary.keys()):
                
                #multiple DelayObject can have same radix value therefore a list is used
                object_dictionary[delay_object.radix]=[]
            
            # appending the DelayObject to the list corresponding to the respecitve radix 
            object_dictionary[delay_object.radix].append(delay_object)
            
        if current_line == "$upscope $end":
            end_of_file = True
        current_line = f.readline().rstrip("\n") 
        
except KeyboardInterrupt:
    pass



# dumpvars checkpoint
while current_line != "$dumpvars":
    current_line=f.readline().rstrip("\n")
    
# ends checkpoint
while current_line != "$end":
    current_line=f.readline().rstrip("\n")
    
    
current_line = f.readline().rstrip("\n")

#time of switching // current time
time = ""

while current_line != "":
    
    #check if the current line contains the time
    if current_line[0] == "#":
        
        #read the time of format: "#<time>"
        time = float(current_line[1:]) * 1e-8
        
    #checking if the signal is binary?
    elif current_line[0] == "b" or current_line[0] == "B":
        
        
        #splitting the string containing signal value and the identifier
        current_line_split_list = current_line.split() 
        
        #storing the time in the dictionary with time as key and value 0 or 1 as the value
        radix = current_line_split_list[-1]
        
        #object list corresponding to the given radix
        object_list= object_dictionary[radix]
        
        #assigning value to the binary-DelayObject
        for i in range(len(object_list)):
            
            binary_object = object_list[i]
            toggle_value = 1 if binary_object.binary_flag==0 else 0
            binary_object.toggle[time] = toggle_value
            binary_object.binary_flag = toggle_value
            #the delay object has a long binary output    
            binary_object.binary = True
            
    #else just store the value as 0th index of a line and the remaining string as the radix(identifier)
    else:
        
        #retriving radix
        radix = current_line[1:]
        
        #retriving the list of delay objects associated with the radix
        object_list = object_dictionary[radix]
        
        
        for i in range(len(object_list)):
            delay_object = object_list[i]
            delay_object.toggle[time] = int(current_line[0])
    
    current_line = f.readline().rstrip("\n")

    
# Assuming object_dictionary is defined and populated in parsingfile.ipynb
with open('../newPickle/object_dictionary.pickle', 'wb') as f:
    pickle.dump(object_dictionary, f)