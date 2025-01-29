import random
import sys
from tqdm import tqdm  # Import tqdm

def genLinear():
    x = random.randint(-100,100)
    m = random.randint(-100,100)
    b = random.randint(-100,100)
    express = ""
    if (b>0):
        express = f"{m}x+{b}"
    elif (b==0):
        express = f"{m}x"
    elif (b<0):
        express = f"{m}x{b}"
    while m == 0:
        m = random.randint(-100,100)
    sol = m*x+b
    return f"<SFX>{express}={sol}<SOL>x={x}"

if __name__ == "__main__":
    if len(sys.argv) > 1:
        try: 
            lines = int(sys.argv[1])
        except:
            print(f"Not a valid argument, must be an integer.")
            exit(1)
        print(f"Generating {lines} line(s)")
    else:
        print("No valid arguments passed.")
        exit(1)
    
    # Use tqdm to show progress
    with open("data.txt", 'a') as file:
        for i in tqdm(range(lines), desc="Generating lines", unit="line"):
            file.write(f"{genLinear()}\n")
