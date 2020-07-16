import numpy as np
import PIL.Image, PIL.ImageDraw, PIL.ImageFont

default_font = PIL.ImageFont.truetype('/usr/share/fonts/noto/NotoSans-Regular.ttf',32)
def render_text(txt,origin=(2,-1),size=(256,16),render_scale=1,font=default_font,bg=0,fg=1):
    img = PIL.Image.new('L', (size[0]*render_scale,size[1]*render_scale), bg)
    draw = PIL.ImageDraw.Draw(img)
    draw.text((origin[0]*render_scale,origin[1]*render_scale), txt, fg, font=font)
    img = img.resize(size, PIL.Image.ANTIALIAS)
    return np.asarray(img.getdata()).reshape((size[1],size[0]))

approx_freq = {
    " ":1000,
    '.':200,
    ',':200,
    '!':200,
    '?':200,
    "a":29,
    "c":16,
    "b":3,
    "e":37,
    "d":18,
    "g":3,
    "f":3,
    "i":42,
    "h":1,
    "m":17,
    "l":21,
    "o":29,
    "n":24,
    "q":5,
    "p":11,
    "s":18,
    "r":22,
    "u":28,
    "t":32,
    "v":3,
    "x":3,
    "w":3,
    "y":3,
    "A":29,
    "C":16,
    "B":3,
    "E":37,
    "D":18,
    "G":3,
    "F":3,
    "I":42,
    "H":1,
    "M":17,
    "L":21,
    "O":29,
    "N":24,
    "Q":5,
    "P":11,
    "S":18,
    "R":22,
    "U":28,
    "T":32,
    "V":3,
    "X":3,
    "W":3,
    "Y":3
}
import string
for letter in string.ascii_lowercase:
    if letter in approx_freq:
        approx_freq[letter] *=  10

letters = np.asarray([chr(c) for c in range(ord('A'),ord('Z')+1)])
#letters = np.asarray(list(approx_freq.keys()))
indices = {s:i for i,s in enumerate(letters)}

import random 
if hasattr(random,'choices'):
    choices = random.choices
else:
    def choices(seq,weights=None,k=1):
        if weights is None:
            return [random.choice(seq) for i in range(k)]
        else:
            weights = np.asarray(weights)
            cdf = np.cumsum(weights)
            np.insert(cdf,0,0)
            rnd = np.random.randint(cdf[-1],size=k)
            idx = np.digitize(rnd,cdf)
            return list(np.asarray(seq)[idx])
        
letters_freq = {letter:(approx_freq[letter] if letter in approx_freq else 1) for letter in letters}
def speech_like(i,j):
    vals = np.asarray(list(letters_freq.keys()))
    w = np.asarray(list(letters_freq.values()))
    txt_block = np.asarray(choices(vals,weights=w,k=i*j)).reshape((i,j))
    return txt_block

def gen_data(txt,width=32,height=32,font=default_font,flat=False,shuffle=True):
    if shuffle:
        origin = (np.random.randint(0,6),-np.random.randint(6,10))
    else:
        origin = (0,-4)
    img = render_text(txt,origin=origin,size=(width,height),font=font)
    truth = np.zeros(out_size)
    truth[indices[txt[0]]] = 1.0
    if flat:
        return truth,img.flatten()
    else:
        return truth,img
    
def tagged_data(n):
    txt_block = speech_like(n,2)
    yield from (gen_data(''.join(i),flat=True) for i in txt_block)
    
def tagged_2d_data(n):
    txt_block = speech_like(n,2)
    yield from (gen_data(''.join(i)) for i in txt_block)
    
out_size = len(letters)
out_shape = (len(letters),)
in_size = len(next(tagged_data(1))[1])
in_2d_shape = next(tagged_2d_data(1))[1].shape