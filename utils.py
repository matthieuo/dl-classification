from pathlib import Path 

class file_to_label_binary:
    def __init__(self): 
        self.classes = {}
        self.curr_class = 0
        
    def to_label(self,in_file):
        p = Path(in_file)
   
        f_dir = p.parts[-2]

        if f_dir not in self.classes:
            self.classes[f_dir] = self.curr_class
            self.curr_class += 1
            
            #assert self.curr_class < 3,"For now, only two classes are supported (binary classification)"
  
        return self.classes[f_dir]
    
  
