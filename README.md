# SmoothLifeSDL
Generalization of Conway's "Game of Life" to a continuous domain, linux implementation with SDL

This is a copy from

# Install SDL and GL development libs [rpm world]

```bash
sudo dnf install SDL2-devel SDL2 SDL_ttf-devel SDL_ttf GLee GLee-devel
```

# Compile
```bash
gcc main.cpp -lSDL -lpthread -Lglut -lGL -lGLU -lm -lSDL_ttf -o smooth
```

# Parameters

```
esc         Quit

x/y         color scheme
X/Y         phase for color schemes 1 and 7
C/V         visualization style 0-6 (3D only)
v           show/hide timing information and values
c           show kernels and snm (switch to mode 0 for correct display)
p           pause
b/n/space   fill buffer with random blobs
m           save values (append at the end of config file)
,           resize window to client area size 640x480 for video recording
.           maximize to full screen / restore window
-           save buffer as .bmp (2D only)
crsr        scroll around (also pg up/down in 3D)
f1/f2       in 3D: rotate box
f3/f4       auto phase changing speed for color mode 7 and 1
f5/f6/f7    n dimensions 1,2,3

q/a         increase/decrease b1 (with shift factor 10 faster)
w/s         increase/decrease b2
e/d         increase/decrease d1
r/f         increase/decrease d2

t/g         sigmode
z/h         sigtype
u/j         mixtype
i/k         sn
o/l         sm

T/G         radius +/-
Z/H         dt +/-
5-9         new size NN=128,256,512,1024,2048 in 2D NN=32,64,128,256,512 in 3D
0/1/2       mode 0/1/2 (discrete/smooth/smooth2 time stepping)

lmb         in 2D move window in 3D rotate box
wheel       zoom

(/)         set paras to paras from list, paras number -/+
```
