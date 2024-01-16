import inspect
import os
import json
import re
#设置属性装饰器，在网络中进行@
def set_attribute(attr, value):  
    def decorator(fn):  
        setattr(fn, attr, value)  
        return fn  
    return decorator


def get_name(stra):
    match = re.search(r',(.*?)\)', stra)  
    if match:  
        extracted_string = match.group(1)  
        #print(extracted_string)  # 输出: ResidualBlock_source 
        return extracted_string
    else:  
        raise Exception("No match found.")
            
def creat_code(net):
    lines,line_offset=inspect.getsourcelines(net.construct)
    var = get_name(lines[0]).strip()
    strx=f"{var}=({str(lines[1:])},{str(line_offset+1)})"
    return strx
#生成sources.py文件
def creat_source_file(nets,filename="sources.py"):
    strs=[creat_code(net) for net in nets]
    with open(filename, 'w') as f:  
        for item in strs:  
            f.write("%s\n" % item)
