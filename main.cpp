/*
        SmoothLife


        esc			Quit

        x/y			color scheme
        X/Y			phase for color schemes 1 and 7
        C/V			visualization style 0-6 (3D only)
        v			show/hide timing information and values
        c			show kernels and snm (switch to mode 0 for
   correct display) p			pause b/n/space	fill buffer with random
   blobs m			save values (append at the end of config file)
        ,			resize window to client area size 640x480 for
   video recording .			maximize to full screen / restore window
        -			save buffer as .bmp (2D only)
        crsr		scroll around (also pg up/down in 3D)
        f1/f2		in 3D: rotate box
        f3/f4		auto phase changing speed for color mode 7 and 1
        f5/f6/f7	n dimensions 1,2,3

        q/a			increase/decrease b1 (with shift factor 10
   faster) w/s			increase/decrease b2 e/d
   increase/decrease d1 r/f			increase/decrease d2

        t/g			sigmode
        z/h			sigtype
        u/j			mixtype
        i/k			sn
        o/l			sm

        T/G			radius +/-
        Z/H			dt +/-
        5-9			new size NN=128,256,512,1024,2048 in 2D
   NN=32,64,128,256,512 in 3D
        0/1/2		mode 0/1/2 (discrete/smooth/smooth2 time stepping)

        lmb			in 2D move window in 3D rotate box
        wheel		zoom

        (/)			set paras to paras from list, paras number -/+
*/

#include <SDL/SDL.h>
#include <SDL/SDL_ttf.h>
#include <fcntl.h>
#include <math.h>
#include <time.h>


#define GL_GLEXT_PROTOTYPES
#include <GL/gl.h>
#include <GL/glu.h>

const double PI =
    6.28318530718;  // circle constant, relation circumference to radius

int dims;  // n dimensions 1, 2 or 3

int NX, NY, NZ;  // buffer size (must be power of 2)
int BX, BY, BZ;  // buffer size power of 2

const int BMAX = 16;  // max power of 2 of buffer size (for plan[] arrays)

// real buffers
const int AA = 0;   // the buffer
const int KR = 1;   // ring kernel
const int KD = 2;   // disk kernel
const int AN = 3;   // buffer blured with ring kernel
const int AM = 4;   // buffer blured with disk kernel
const int ARB = 5;  // number of real buffers

// Fourier buffers (real and imag part, but half the size in x dimension)
const int AF = 0;    // FT of buffer
const int KRF = 1;   // FT of ring kernel
const int KDF = 2;   // FT of disk kernel
const int ANF = 3;   // FT of buffer blured with ring kernel
const int AMF = 4;   // FT of buffer blured with disk kernel
const int FFT0 = 5;  // intermediate FFT buffers (toggle between them)
const int FFT1 = 6;
const int AFB = 7;  // number of Fourier buffers

double kflr, kfld;  // computed areas of disk and ring kernels

const char *prgname = "SmoothLife";
FILE *logfile;
int maximized;            // is the window maximized?
int SX, SY;               // window size
int DX, DY;               // desktop size for fullscreen
unsigned long tim, tima;  // for performance timer

int timing;          // show timing on/off
int oldSX, oldSY;    // window size and position for restoring it
int windowfocus;     // does the window have the focus?
double savedispcnt;  // seconds to display the save confirmation
int anz;             // buffer view mode 1-4
int pause;           // pause toggle

int ox, oy;                 // offset of drawn buffers rectangle in the window
int qx, qy;                 // size of drawn buffer rectangle
int qq;                     // determines zoom for 3D
double fx, fy, fz;          // offset in texture (crsr keys)
double wi, wj, wx, wy, dw;  // viewing angles for 3D and auto rotation speed

double ra, rr, rb, dt;  // outer radius, relation outer/inner, relation
                        // outer/width of anti aliasing zone, time step
double b1, d1, b2, d2;  // birth and death interval boundaries
double sn, sm;  // snm function smoothing step width in n and m direction
int mode;       // time step mode 0 = discrete, 1,2 = smooth
int sigmode;    // sigmoid construction mode 1-4
int sigtype;    // sigmoid type 0-9 in n direction
int mixtype;    // sigmoid type 0-7 in m direction

double colscheme;  // color scheme 1-7
double phase;      // phase for color scheme 1 and 7
double dphase;     // phase changing speed
double visscheme;  // for 3D visualization scheme

GLuint shader_snm, shader_fft, shader_kernelmul, shader_draw;  // shaders
GLuint shader_copybufferrc, shader_copybuffercr;
GLuint fb[AFB], tb[AFB];  // Fourier framebuffers and textures
GLuint fr[ARB], tr[ARB];  // real framebuffers and textures
GLuint planx[BMAX][2], plany[BMAX][2],
    planz[BMAX][2];  // plan 1D textures for FFT
GLuint spfb, sptb;   // buffers for save picture
GLenum ttd;          // texture target dimension depending on 1D, 2D, 3D

GLint loc_b1, loc_b2;  // shader variable locations
GLint loc_d1, loc_d2;
GLint loc_sn, loc_sm;
GLint loc_mode, loc_sigmode, loc_sigtype, loc_mixtype, loc_dt;
GLint loc_colscheme, loc_phase, loc_visscheme;
GLint loc_dim, loc_tang, loc_tangsc, loc_sc;

bool neu;      // new buffer size
bool neuedim;  // new paras

const int DESCSIZE = 64;
struct parameterlist  // list with all parameter lines from the config file
{
  int dims, mode;
  double ra, rr, rb, dt;
  double b1, d1, b2, d2;
  int sigmode, sigtype, mixtype;
  double sn, sm;
  char desc[DESCSIZE];  // description text
};
struct parameterlist paralist[1000];  // parameter list, max 1000 entries
int nparas;                           // n paras in list
int curparas;                         // current paras number in list

char dispmessage[128];  // message to display in 3rd line
int ypos;               // y position for 1D drawing

GLuint fonttex[256];         // for drawtext
int fontw[256], fonth[256];  // size of each character

// return random floating point number 0 <= n < x
//
double RND(double x) { return x * (double)rand() / ((double)RAND_MAX + 1); }

// read all paras from config file into paraslist
//
bool read_config(void) {
  FILE *file;
  int l, t, d;
  char buf[256], desc[256];
  bool gef;

  file = fopen("SmoothLifeConfig.txt", "r");
  if (file == 0) return false;

  l = 0;  // read lines
  for (;;) {
    char *bla = fgets(buf, 256, file);
    if (feof(file)) break;

    if (!(buf[0] == '1' || buf[0] == '2' || buf[0] == '3')) continue;

    sscanf(buf, "%d %d  %lf %lf %lf %lf  %lf %lf %lf %lf  %d %d %d  %lf %lf",
           &dims, &mode, &ra, &rr, &rb, &dt, &b1, &b2, &d1, &d2, &sigmode,
           &sigtype, &mixtype, &sn, &sm);

    t = 0;  // read description
    d = 0;
    gef = false;
    do {
      if (buf[t] == '/') gef = true;
      if (d < DESCSIZE - 2 && gef) desc[d++] = buf[t];
    } while (buf[t++] != '\0');
    desc[d] = '\0';

    paralist[l].dims = dims;
    paralist[l].mode = mode;
    paralist[l].ra = ra;
    paralist[l].rr = rr;
    paralist[l].rb = rb;
    paralist[l].dt = dt;
    paralist[l].b1 = b1;
    paralist[l].b2 = b2;
    paralist[l].d1 = d1;
    paralist[l].d2 = d2;
    paralist[l].sigmode = sigmode;
    paralist[l].sigtype = sigtype;
    paralist[l].mixtype = mixtype;
    paralist[l].sn = sn;
    paralist[l].sm = sm;
    strcpy(paralist[l].desc, desc);

    l++;
  }

  nparas = l;

  fclose(file);

  fprintf(logfile, "%d parameter lines read from config file\n", nparas);
  fflush(logfile);

  return true;
}

// append paras to end of config file
//
bool write_config(void) {
  FILE *file;

  file = fopen("SmoothLifeConfig.txt", "a");
  if (file == 0) return false;

  fprintf(file, "%d ", (int)dims);
  fprintf(file, "%d   ", (int)mode);

  fprintf(file, "%.1f  ", ra);
  fprintf(file, "%.1f  ", rr);
  fprintf(file, "%.1f  ", rb);
  fprintf(file, "%.3f   ", dt);

  fprintf(file, "%.3f  ", b1);
  fprintf(file, "%.3f  ", b2);
  fprintf(file, "%.3f  ", d1);
  fprintf(file, "%.3f   ", d2);

  fprintf(file, "%d ", (int)sigmode);
  fprintf(file, "%d ", (int)sigtype);
  fprintf(file, "%d   ", (int)mixtype);

  fprintf(file, "%.3f  ", sn);
  fprintf(file, "%.3f    //\n", sm);

  fclose(file);
  return true;
}

// set paras to paras number l from list
//
void setparas(int l) {
  if (l >= 0 && l < nparas) {
    dims = paralist[l].dims;
    mode = paralist[l].mode;
    ra = paralist[l].ra;
    rr = paralist[l].rr;
    rb = paralist[l].rb;
    dt = paralist[l].dt;
    b1 = paralist[l].b1;
    b2 = paralist[l].b2;
    d1 = paralist[l].d1;
    d2 = paralist[l].d2;
    sigmode = paralist[l].sigmode;
    sigtype = paralist[l].sigtype;
    mixtype = paralist[l].mixtype;
    sn = paralist[l].sn;
    sm = paralist[l].sm;
  }
}

// make a buffer for mysavepic
//
bool create_render_buffer(void) {
  unsigned int err;

  glGenTextures(1, &sptb);
  err = glGetError();
  fprintf(logfile, "sp GenTextures err %d\n", err);
  fflush(logfile);
  if (err) return false;

  glBindTexture(GL_TEXTURE_2D, sptb);
  err = glGetError();
  fprintf(logfile, "sp BindTexture err %d\n", err);
  fflush(logfile);
  if (err) return false;

  glTexImage2D(GL_TEXTURE_2D, 0, GL_RGB8, NX, NY, 0, GL_RGB, GL_UNSIGNED_BYTE,
               NULL);
  err = glGetError();
  fprintf(logfile, "sp TexImage %d %d  err %d\n", NX, NY, err);
  fflush(logfile);
  if (err) return false;

  glGenFramebuffers(1, &spfb);
  err = glGetError();
  fprintf(logfile, "sp GenFramebuffers err %d\n", err);
  fflush(logfile);
  if (err) return false;

  glBindFramebuffer(GL_FRAMEBUFFER, spfb);
  err = glGetError();
  fprintf(logfile, "sp BindFramebuffer err %d\n", err);
  fflush(logfile);
  if (err) return false;

  glFramebufferTexture2D(GL_FRAMEBUFFER, GL_COLOR_ATTACHMENT0, GL_TEXTURE_2D,
                         sptb, 0);
  err = glGetError();
  fprintf(logfile, "sp FramebufferTexture2D err %d\n", err);
  fflush(logfile);
  if (err) return false;

  err = glCheckFramebufferStatus(GL_FRAMEBUFFER);
  fprintf(logfile, "sp FramebufferStatus 0x%x\n", err);
  fflush(logfile);
  if (err != GL_FRAMEBUFFER_COMPLETE) return false;

  fprintf(logfile, "\n");
  fflush(logfile);
  return true;
}

// delete mysavepic buffer
//
bool delete_render_buffer(void) {
  unsigned int err;

  glBindTexture(GL_TEXTURE_2D, 0);
  err = glGetError();
  fprintf(logfile, "sp BindTexture 0 err %d\n", err);
  fflush(logfile);
  if (err) return false;
  glDeleteTextures(1, &sptb);
  err = glGetError();
  fprintf(logfile, "sp DeleteTextures err %d\n", err);
  fflush(logfile);
  if (err) return false;

  glBindFramebuffer(GL_FRAMEBUFFER, 0);
  err = glGetError();
  fprintf(logfile, "sp BindFramebuffer 0 err %d\n", err);
  fflush(logfile);
  if (err) return false;
  glDeleteFramebuffers(1, &spfb);
  err = glGetError();
  fprintf(logfile, "sp DeleteFramebuffers err %d\n", err);
  fflush(logfile);
  if (err) return false;

  fprintf(logfile, "\n");
  fflush(logfile);
  return true;
}

// draw buffer for mysavepic
//
void drawa_render_buffer(int a) {
  glMatrixMode(GL_PROJECTION);
  glLoadIdentity();
  glOrtho(0, NX, 0, NY, -1, 1);
  glViewport(0, 0, NX, NY);

  glBindFramebuffer(GL_FRAMEBUFFER, spfb);
  glUseProgram(shader_draw);
  glUniform1f(loc_colscheme, (float)colscheme);
  glUniform1f(loc_phase, (float)phase);

  glClearColor(0.5, 0.5, 0.5, 1.0);
  glClear(GL_COLOR_BUFFER_BIT);

  glActiveTexture(GL_TEXTURE0);
  glBindTexture(GL_TEXTURE_2D, tr[a]);
  glUniform1i(glGetUniformLocation(shader_draw, "tex0"), 0);

  glBegin(GL_QUADS);
  glMultiTexCoord2d(GL_TEXTURE0, fx, fy + 1);
  glVertex2i(0, 0);  // upside down for bmp format
  glMultiTexCoord2d(GL_TEXTURE0, fx + 1, fy + 1);
  glVertex2i(NX, 0);
  glMultiTexCoord2d(GL_TEXTURE0, fx + 1, fy);
  glVertex2i(NX, NY);
  glMultiTexCoord2d(GL_TEXTURE0, fx, fy);
  glVertex2i(0, NY);
  glEnd();

  glUseProgram(0);
}

// if in 2D, save whole buffer as .bmp
//
void mysavepic(void) {
  static int bildnr = 1;
  char filename[128];
  char *buffer;
  SDL_Surface *surf;

  create_render_buffer();
  drawa_render_buffer(AA);

  buffer = (char *)calloc(3 * NX * NY, sizeof(char));
  glReadPixels(0, 0, NX, NY, GL_BGR, GL_UNSIGNED_BYTE, buffer);

  surf = SDL_CreateRGBSurfaceFrom(buffer, NX, NY, 24, NX * 3, 0xff0000,
                                  0x00ff00, 0x0000ff, 0);

  sprintf(filename, "SmoothLife%02d.bmp", bildnr);
  SDL_SaveBMP(surf, filename);

  SDL_FreeSurface(surf);
  free(buffer);
  delete_render_buffer();

  bildnr++;
}

// print shader info log to logfile
//
bool printShaderInfoLog(GLuint obj) {
  int infologLength = 0;
  int charsWritten = 0;
  char *infoLog;

  glGetShaderiv(obj, GL_INFO_LOG_LENGTH, &infologLength);

  if (infologLength > 0) {
    infoLog = (char *)malloc(infologLength);
    glGetShaderInfoLog(obj, infologLength, &charsWritten, infoLog);
    fprintf(logfile, "%s\n", infoLog);
    fflush(logfile);
    free(infoLog);
  }
  return infologLength > 2;
}

// print program info log to logfile
//
bool printProgramInfoLog(GLuint obj) {
  int infologLength = 0;
  int charsWritten = 0;
  char *infoLog;

  glGetProgramiv(obj, GL_INFO_LOG_LENGTH, &infologLength);

  if (infologLength > 0) {
    infoLog = (char *)malloc(infologLength);
    glGetProgramInfoLog(obj, infologLength, &charsWritten, infoLog);
    fprintf(logfile, "%s\n", infoLog);
    fflush(logfile);
    free(infoLog);
  }
  return infologLength > 2;
}

// read .vert and .frag shaders from the files in the shader directory
//
bool setShaders(int dim, char *fname, GLuint &prog) {
  char *vs = NULL, *fs = NULL, *gs = NULL;

  FILE *fp;
  int count = 0;
  char filename[128];

  fprintf(logfile, "setting shader: %s\n", fname);

  sprintf(filename, "shaders/%s%dD.vert", fname, dim);
  fp = fopen(filename, "rt");
  if (fp == NULL) {
    fprintf(logfile, "couldn't open .vert\n");
    fflush(logfile);
    return false;
  }
  fseek(fp, 0, SEEK_END);
  count = ftell(fp);
  rewind(fp);
  if (count > 0) {
    vs = (char *)malloc(sizeof(char) * (count + 1));
    count = fread(vs, sizeof(char), count, fp);
    vs[count] = '\0';
  }
  fclose(fp);

  sprintf(filename, "shaders/%s%dD.frag", fname, dim);
  fp = fopen(filename, "rt");
  if (fp == NULL) {
    fprintf(logfile, "couldn't open .frag\n");
    fflush(logfile);
    return false;
  }
  fseek(fp, 0, SEEK_END);
  count = ftell(fp);
  rewind(fp);
  if (count > 0) {
    fs = (char *)malloc(sizeof(char) * (count + 1));
    count = fread(fs, sizeof(char), count, fp);
    fs[count] = '\0';
  }
  fclose(fp);

  // insert constants into the shader source code (not used)
  char w4[256];
  int t, d, w;

  t = 0;
  while (fs[t] != '\0') t++;

  gs = (char *)calloc(t * 2, sizeof(char));

  t = 0;
  d = 0;
  w4[0] = '\0';

  do {
    w = 0;
    while (!(fs[t] == ' ' || fs[t] == '\n' || fs[t] == '\0')) {
      w4[w] = fs[t];
      w++;
      t++;
    }
    w4[w] = '\0';

    w = 0;
    while (w4[w] != '\0') {
      gs[d] = w4[w];
      d++;
      w++;
    }

    while (fs[t] == ' ' || fs[t] == '\n' || fs[t] == '\0') {
      gs[d] = fs[t];
      if (fs[t] == '\0') break;
      d++;
      t++;
    }
  } while (fs[t] != '\0');

  /*FILE *file;
  sprintf (filename, "%s.test", fname);
  file = fopen (filename, "w");
  fwrite (gs, 1, d, file);
  fclose (file);*/

  GLuint v, f;
  v = glCreateShader(GL_VERTEX_SHADER);
  f = glCreateShader(GL_FRAGMENT_SHADER);

  const char *vv = vs;
  const char *ff = gs;

  glShaderSource(v, 1, &vv, NULL);
  glShaderSource(f, 1, &ff, NULL);

  free(vs);
  free(fs);
  free(gs);

  bool ret;
  ret = false;

  glCompileShader(v);
  if (printShaderInfoLog(v)) {
    fprintf(logfile, "error in vertex shader!\n\n");
    fflush(logfile);
    ret |= true;
  } else {
    fprintf(logfile, "vertex shader ok\n\n");
    fflush(logfile);
  }

  glCompileShader(f);
  if (printShaderInfoLog(f)) {
    fprintf(logfile, "error in fragment shader!\n\n");
    fflush(logfile);
    ret |= true;
  } else {
    fprintf(logfile, "fragment shader ok\n\n");
    fflush(logfile);
  }

  prog = glCreateProgram();
  glAttachShader(prog, v);
  glAttachShader(prog, f);

  glLinkProgram(prog);
  if (printProgramInfoLog(prog)) {
    fprintf(logfile, "shader program error!\n\n");
    fflush(logfile);
    ret |= true;
  } else {
    fprintf(logfile, "shader program ok\n\n");
    fflush(logfile);
  }

  // return ret;		// doesn't work with ati, so...
  return 0;
}

// delete shader programs (doesn't delete shaders, but whatever)
//
void delShaders(void) {
  unsigned int err;

  fprintf(logfile, "delete shaders\n");
  fflush(logfile);

  glUseProgram(0);
  err = glGetError();
  fprintf(logfile, "UseProgram 0 err %d\n", err);
  fflush(logfile);

  glDeleteProgram(shader_copybuffercr);
  err = glGetError();
  fprintf(logfile, "DeleteProgram copybuffercr err %d\n", err);
  fflush(logfile);

  glDeleteProgram(shader_copybufferrc);
  err = glGetError();
  fprintf(logfile, "DeleteProgram copybufferrc err %d\n", err);
  fflush(logfile);

  glDeleteProgram(shader_draw);
  err = glGetError();
  fprintf(logfile, "DeleteProgram draw err %d\n", err);
  fflush(logfile);

  glDeleteProgram(shader_fft);
  err = glGetError();
  fprintf(logfile, "DeleteProgram fft err %d\n", err);
  fflush(logfile);

  glDeleteProgram(shader_kernelmul);
  err = glGetError();
  fprintf(logfile, "DeleteProgram kernelmul err %d\n", err);
  fflush(logfile);

  glDeleteProgram(shader_snm);
  err = glGetError();
  fprintf(logfile, "DeleteProgram snm err %d\n", err);
  fflush(logfile);
}

// put this in all tight nested for-loops to stay responsive
//
void dosystem(void) {
  // SDL_Event event;
  // SDL_PollEvent (&event);
}

// put a 3D splat in buf
//
void splat3D(float *buf) {
  double mx, my, mz, dx, dy, dz, u, l;
  int ix, iy, iz;

  mx = RND(NX);
  my = RND(NY);
  mz = RND(NZ);
  u = ra * (RND(0.5) + 0.5);

  for (iz = (int)(mz - u - 1); iz <= (int)(mz + u + 1); iz++)
    for (iy = (int)(my - u - 1); iy <= (int)(my + u + 1); iy++)
      for (ix = (int)(mx - u - 1); ix <= (int)(mx + u + 1); ix++) {
        dx = mx - ix;
        dy = my - iy;
        dz = mz - iz;
        l = sqrt(dx * dx + dy * dy + dz * dz);
        if (l < u) {
          int px = ix;
          int py = iy;
          int pz = iz;
          while (px < 0) px += NX;
          while (px >= NX) px -= NX;
          while (py < 0) py += NY;
          while (py >= NY) py -= NY;
          while (pz < 0) pz += NZ;
          while (pz >= NZ) pz -= NZ;
          if (px >= 0 && px < NX && py >= 0 && py < NY && pz >= 0 && pz < NZ) {
            *(buf + NX * NY * pz + NX * py + px) = 1.0;
          }
        }
        dosystem();
      }
}

// init 3D buffer with splats
//
void inita3D(int a) {
  float *buf = (float *)calloc(NX * NY * NZ, sizeof(float));

  double mx, my, mz;

  mx = 2 * ra;
  if (mx > NX) mx = NX;
  my = 2 * ra;
  if (my > NY) my = NY;
  mz = 2 * ra;
  if (mz > NZ) mz = NZ;

  for (int t = 0; t <= (int)(NX * NY * NZ / (mx * my * mz)); t++) {
    splat3D(buf);
  }

  glBindTexture(GL_TEXTURE_3D, tr[a]);
  glTexImage3D(GL_TEXTURE_3D, 0, GL_R32F, NX, NY, NZ, 0, GL_RED, GL_FLOAT, buf);

  free(buf);
}

// put a 2D splat in buf
//
void splat2D(float *buf) {
  double mx, my, dx, dy, u, l;
  int ix, iy;

  mx = RND(NX);
  my = RND(NY);
  u = ra * (RND(0.5) + 0.5);

  for (iy = (int)(my - u - 1); iy <= (int)(my + u + 1); iy++)
    for (ix = (int)(mx - u - 1); ix <= (int)(mx + u + 1); ix++) {
      dx = mx - ix;
      dy = my - iy;
      l = sqrt(dx * dx + dy * dy);
      if (l < u) {
        int px = ix;
        int py = iy;
        while (px < 0) px += NX;
        while (px >= NX) px -= NX;
        while (py < 0) py += NY;
        while (py >= NY) py -= NY;
        if (px >= 0 && px < NX && py >= 0 && py < NY) {
          *(buf + NX * py + px) = 1.0;
        }
      }
      dosystem();
    }
}

// init 2D buffer with splats
//
void inita2D(int a) {
  float *buf = (float *)calloc(NX * NY, sizeof(float));

  double mx, my;

  mx = 2 * ra;
  if (mx > NX) mx = NX;
  my = 2 * ra;
  if (my > NY) my = NY;

  for (int t = 0; t <= (int)(NX * NY / (mx * my)); t++) {
    splat2D(buf);
  }

  glBindTexture(GL_TEXTURE_2D, tr[a]);
  glTexImage2D(GL_TEXTURE_2D, 0, GL_R32F, NX, NY, 0, GL_RED, GL_FLOAT, buf);

  free(buf);
}

// put a 1D splat in buf
//
void splat1D(float *buf) {
  double mx, dx, u, l;
  int ix;

  mx = RND(NX);
  u = ra * (RND(0.5) + 0.5);

  for (ix = (int)(mx - u - 1); ix <= (int)(mx + u + 1); ix++) {
    dx = mx - ix;
    l = sqrt(dx * dx);
    if (l < u) {
      int px = ix;
      while (px < 0) px += NX;
      while (px >= NX) px -= NX;
      if (px >= 0 && px < NX) {
        *(buf + px) = 1.0;
      }
    }
    dosystem();
  }
}

// init a 1D buffer with splats
//
void inita1D(int a) {
  float *buf = (float *)calloc(NX, sizeof(float));

  double mx;

  mx = 2 * ra;
  if (mx > NX) mx = NX;

  for (int t = 0; t <= (int)(NX / mx); t++) {
    splat1D(buf);
  }

  glBindTexture(GL_TEXTURE_1D, tr[a]);
  glTexImage1D(GL_TEXTURE_1D, 0, GL_R32F, NX, 0, GL_RED, GL_FLOAT, buf);

  free(buf);
}

// init buffer with splats
//
void inita(int a) {
  if (dims == 1) inita1D(a);
  if (dims == 2) inita2D(a);
  if (dims == 3) inita3D(a);
}

// create real and Fourier buffers
//
bool create_buffers(void) {
  unsigned int err;
  int t, s, eb;

  if (dims == 1) {
    fprintf(logfile, "create buffers 1D %d\n", NX);
    fflush(logfile);
  } else if (dims == 2) {
    fprintf(logfile, "create buffers 2D %d %d\n", NX, NY);
    fflush(logfile);
  } else  // dims==3
  {
    fprintf(logfile, "create buffers 3D %d %d %d\n", NX, NY, NZ);
    fflush(logfile);
  }

  // Fourier (complex) buffers

  glGenTextures(AFB, &tb[0]);
  err = glGetError();
  fprintf(logfile, "GenTextures err %d\n", err);
  fflush(logfile);
  if (err) return false;

  glGenFramebuffers(AFB, &fb[0]);
  err = glGetError();
  fprintf(logfile, "GenFramebuffers err %d\n", err);
  fflush(logfile);
  if (err) return false;

  for (t = 0; t < AFB; t++) {
    fprintf(logfile, "complex buffer %d\n", t);
    fflush(logfile);

    glBindTexture(ttd, tb[t]);
    err = glGetError();
    fprintf(logfile, "BindTexture err %d\n", err);
    fflush(logfile);
    if (err) return false;

    glTexParameterf(ttd, GL_TEXTURE_MIN_FILTER, GL_NEAREST);
    glTexParameterf(ttd, GL_TEXTURE_MAG_FILTER, GL_NEAREST);
    glTexParameterf(ttd, GL_TEXTURE_WRAP_S, GL_REPEAT);
    if (dims > 1) {
      glTexParameterf(ttd, GL_TEXTURE_WRAP_T, GL_REPEAT);
    }
    if (dims > 2) {
      glTexParameterf(ttd, GL_TEXTURE_WRAP_R, GL_REPEAT);
    }

    if (dims == 1)
      glTexImage1D(GL_TEXTURE_1D, 0, GL_RG32F, NX / 2 + 1, 0, GL_RG, GL_FLOAT,
                   NULL);
    if (dims == 2)
      glTexImage2D(GL_TEXTURE_2D, 0, GL_RG32F, NX / 2 + 1, NY, 0, GL_RG,
                   GL_FLOAT, NULL);
    if (dims == 3)
      glTexImage3D(GL_TEXTURE_3D, 0, GL_RG32F, NX / 2 + 1, NY, NZ, 0, GL_RG,
                   GL_FLOAT, NULL);
    err = glGetError();
    fprintf(logfile, "TexImage err %d\n", err);
    fflush(logfile);
    if (err) return false;

    glBindFramebuffer(GL_FRAMEBUFFER, fb[t]);
    err = glGetError();
    fprintf(logfile, "BindFramebuffer err %d\n", err);
    fflush(logfile);
    if (err) return false;

    if (dims == 1)
      glFramebufferTexture1D(GL_FRAMEBUFFER, GL_COLOR_ATTACHMENT0,
                             GL_TEXTURE_1D, tb[t], 0);
    if (dims == 2)
      glFramebufferTexture2D(GL_FRAMEBUFFER, GL_COLOR_ATTACHMENT0,
                             GL_TEXTURE_2D, tb[t], 0);
    if (dims == 3)
      glFramebufferTexture3D(GL_FRAMEBUFFER, GL_COLOR_ATTACHMENT0,
                             GL_TEXTURE_3D, tb[t], 0, 0);
    err = glGetError();
    fprintf(logfile, "FramebufferTexture err %d\n", err);
    fflush(logfile);
    if (err) return false;

    err = glCheckFramebufferStatus(GL_FRAMEBUFFER);
    fprintf(logfile, "FramebufferStatus 0x%x\n", err);
    fflush(logfile);
    if (err != GL_FRAMEBUFFER_COMPLETE) return false;
  }

  // real buffers

  glGenTextures(ARB, &tr[0]);
  err = glGetError();
  fprintf(logfile, "GenTextures err %d\n", err);
  fflush(logfile);
  if (err) return false;

  glGenFramebuffers(ARB, &fr[0]);
  err = glGetError();
  fprintf(logfile, "GenFramebuffers err %d\n", err);
  fflush(logfile);
  if (err) return false;

  for (t = 0; t < ARB; t++) {
    fprintf(logfile, "real buffer %d\n", t);
    fflush(logfile);

    glBindTexture(ttd, tr[t]);
    err = glGetError();
    fprintf(logfile, "BindTexture err %d\n", err);
    fflush(logfile);
    if (err) return false;

    glTexParameterf(ttd, GL_TEXTURE_MIN_FILTER, GL_NEAREST);
    glTexParameterf(ttd, GL_TEXTURE_MAG_FILTER, GL_NEAREST);
    glTexParameterf(ttd, GL_TEXTURE_WRAP_S, GL_REPEAT);
    if (dims > 1) {
      glTexParameterf(ttd, GL_TEXTURE_WRAP_T, GL_REPEAT);
    }
    if (dims > 2) {
      glTexParameterf(ttd, GL_TEXTURE_WRAP_R, GL_REPEAT);
    }

    if (dims == 1)
      glTexImage1D(GL_TEXTURE_1D, 0, GL_R32F, NX, 0, GL_RED, GL_FLOAT, NULL);
    if (dims == 2)
      glTexImage2D(GL_TEXTURE_2D, 0, GL_R32F, NX, NY, 0, GL_RED, GL_FLOAT,
                   NULL);
    if (dims == 3)
      glTexImage3D(GL_TEXTURE_3D, 0, GL_R32F, NX, NY, NZ, 0, GL_RED, GL_FLOAT,
                   NULL);
    err = glGetError();
    fprintf(logfile, "TexImage err %d\n", err);
    fflush(logfile);
    if (err) return false;

    glBindFramebuffer(GL_FRAMEBUFFER, fr[t]);
    err = glGetError();
    fprintf(logfile, "BindFramebuffer err %d\n", err);
    fflush(logfile);
    if (err) return false;

    if (dims == 1)
      glFramebufferTexture1D(GL_FRAMEBUFFER, GL_COLOR_ATTACHMENT0,
                             GL_TEXTURE_1D, tr[t], 0);
    if (dims == 2)
      glFramebufferTexture2D(GL_FRAMEBUFFER, GL_COLOR_ATTACHMENT0,
                             GL_TEXTURE_2D, tr[t], 0);
    if (dims == 3)
      glFramebufferTexture3D(GL_FRAMEBUFFER, GL_COLOR_ATTACHMENT0,
                             GL_TEXTURE_3D, tr[t], 0, 0);
    err = glGetError();
    fprintf(logfile, "FramebufferTexture err %d\n", err);
    fflush(logfile);
    if (err) return false;

    err = glCheckFramebufferStatus(GL_FRAMEBUFFER);
    fprintf(logfile, "FramebufferStatus 0x%x\n", err);
    fflush(logfile);
    if (err != GL_FRAMEBUFFER_COMPLETE) return false;
  }

  // FFT plan texture

  glGenTextures((BX - 1 + 2) * 2, &planx[0][0]);
  for (s = 0; s <= 1; s++) {
    for (eb = 0; eb <= BX - 1 + 1; eb++) {
      glBindTexture(GL_TEXTURE_1D, planx[eb][s]);
      err = glGetError();
      fprintf(logfile, "BindTexture err %d\n", err);
      fflush(logfile);
      if (err) return false;

      glTexParameterf(GL_TEXTURE_1D, GL_TEXTURE_MIN_FILTER, GL_NEAREST);
      glTexParameterf(GL_TEXTURE_1D, GL_TEXTURE_MAG_FILTER, GL_NEAREST);
      glTexParameterf(GL_TEXTURE_1D, GL_TEXTURE_WRAP_S, GL_REPEAT);

      glTexImage1D(GL_TEXTURE_1D, 0, GL_RGBA32F, NX / 2 + 1, 0, GL_RGBA,
                   GL_FLOAT, NULL);
      err = glGetError();
      fprintf(logfile, "TexImage err %d\n", err);
      fflush(logfile);
      if (err) return false;
    }
  }

  if (dims > 1) {
    glGenTextures((BY + 1) * 2, &plany[0][0]);
    for (s = 0; s <= 1; s++) {
      for (eb = 1; eb <= BY; eb++) {
        glBindTexture(GL_TEXTURE_1D, plany[eb][s]);
        err = glGetError();
        fprintf(logfile, "BindTexture err %d\n", err);
        fflush(logfile);
        if (err) return false;

        glTexParameterf(GL_TEXTURE_1D, GL_TEXTURE_MIN_FILTER, GL_NEAREST);
        glTexParameterf(GL_TEXTURE_1D, GL_TEXTURE_MAG_FILTER, GL_NEAREST);
        glTexParameterf(GL_TEXTURE_1D, GL_TEXTURE_WRAP_S, GL_REPEAT);

        glTexImage1D(GL_TEXTURE_1D, 0, GL_RGBA32F, NY, 0, GL_RGBA, GL_FLOAT,
                     NULL);
        err = glGetError();
        fprintf(logfile, "TexImage err %d\n", err);
        fflush(logfile);
        if (err) return false;
      }
    }
  }

  if (dims > 2) {
    glGenTextures((BZ + 1) * 2, &planz[0][0]);
    for (s = 0; s <= 1; s++) {
      for (eb = 1; eb <= BZ; eb++) {
        glBindTexture(GL_TEXTURE_1D, planz[eb][s]);
        err = glGetError();
        fprintf(logfile, "BindTexture err %d\n", err);
        fflush(logfile);
        if (err) return false;

        glTexParameterf(GL_TEXTURE_1D, GL_TEXTURE_MIN_FILTER, GL_NEAREST);
        glTexParameterf(GL_TEXTURE_1D, GL_TEXTURE_MAG_FILTER, GL_NEAREST);
        glTexParameterf(GL_TEXTURE_1D, GL_TEXTURE_WRAP_S, GL_REPEAT);

        glTexImage1D(GL_TEXTURE_1D, 0, GL_RGBA32F, NZ, 0, GL_RGBA, GL_FLOAT,
                     NULL);
        err = glGetError();
        fprintf(logfile, "TexImage err %d\n", err);
        fflush(logfile);
        if (err) return false;
      }
    }
  }

  fprintf(logfile, "all buffers ok\n");
  fflush(logfile);

  return true;
}

// delete all buffers
//
void delete_buffers(void) {
  unsigned int err;

  fprintf(logfile, "delete buffers\n");
  fflush(logfile);

  glBindTexture(ttd, 0);
  err = glGetError();
  fprintf(logfile, "BindTexture 0 err %d\n", err);
  fflush(logfile);

  glDeleteTextures(AFB, &tb[0]);
  err = glGetError();
  fprintf(logfile, "DeleteTextures err %d\n", err);
  fflush(logfile);

  glDeleteTextures(ARB, &tr[0]);
  err = glGetError();
  fprintf(logfile, "DeleteTextures err %d\n", err);
  fflush(logfile);

  glDeleteTextures((BX - 1 + 2) * 2, &planx[0][0]);
  err = glGetError();
  fprintf(logfile, "DeleteTextures err %d\n", err);
  fflush(logfile);

  if (dims > 1) {
    glDeleteTextures((BY + 1) * 2, &plany[0][0]);
    err = glGetError();
    fprintf(logfile, "DeleteTextures err %d\n", err);
    fflush(logfile);
  }
  if (dims > 2) {
    glDeleteTextures((BZ + 1) * 2, &planz[0][0]);
    err = glGetError();
    fprintf(logfile, "DeleteTextures err %d\n", err);
    fflush(logfile);
  }

  glBindFramebuffer(GL_FRAMEBUFFER, 0);
  err = glGetError();
  fprintf(logfile, "BindFramebuffer 0 err %d\n", err);
  fflush(logfile);

  glDeleteFramebuffers(AFB, &fb[0]);
  err = glGetError();
  fprintf(logfile, "DeleteFramebuffers err %d\n", err);
  fflush(logfile);

  glDeleteFramebuffers(ARB, &fr[0]);
  err = glGetError();
  fprintf(logfile, "DeleteFramebuffers err %d\n", err);
  fflush(logfile);
}

// coordinates for the 3D cube
//
int cube[6][4][3] = {{{1, 1, -1}, {-1, 1, -1}, {-1, 1, 1}, {1, 1, 1}},
                     {{1, -1, 1}, {-1, -1, 1}, {-1, -1, -1}, {1, -1, -1}},
                     {{1, 1, 1}, {-1, 1, 1}, {-1, -1, 1}, {1, -1, 1}},
                     {{1, -1, -1}, {-1, -1, -1}, {-1, 1, -1}, {1, 1, -1}},
                     {{-1, 1, 1}, {-1, 1, -1}, {-1, -1, -1}, {-1, -1, 1}},
                     {{1, 1, -1}, {1, 1, 1}, {1, -1, 1}, {1, -1, -1}}};

// draw the buffer
//
void drawa(int a) {
  if (dims == 1) {
    glMatrixMode(GL_PROJECTION);
    glLoadIdentity();
    glOrtho(0, SX, 0, SY, -1, 1);
    glViewport(0, 0, SX, SY);

    glBindFramebuffer(GL_FRAMEBUFFER, 0);

    // glClearColor (0.5, 0.5, 0.5, 1.0);
    // glClear (GL_COLOR_BUFFER_BIT);

    glActiveTexture(GL_TEXTURE0);
    glEnable(GL_TEXTURE_1D);
    glBindTexture(GL_TEXTURE_1D, tr[a]);

    glActiveTexture(GL_TEXTURE1);
    glDisable(GL_TEXTURE_1D);

    glUseProgram(shader_draw);
    glUniform1i(glGetUniformLocation(shader_draw, "tex0"), 0);
    glUniform1f(loc_colscheme, (float)colscheme);
    glUniform1f(loc_phase, (float)phase);

    glDisable(GL_DEPTH_TEST);

    glBegin(GL_LINES);
    glMultiTexCoord1d(GL_TEXTURE0, fx);
    glVertex2d(0, ypos + 0.5);
    glMultiTexCoord1d(GL_TEXTURE0, fx + 1);
    glVertex2d(qx, ypos + 0.5);
    glEnd();

    glUseProgram(0);

  } else if (dims == 2) {
    glMatrixMode(GL_PROJECTION);
    glLoadIdentity();
    glOrtho(0, SX, 0, SY, -1, 1);
    glViewport(0, 0, SX, SY);

    glBindFramebuffer(GL_FRAMEBUFFER, 0);

    glClearColor(0.5, 0.5, 0.5, 1.0);
    glClear(GL_COLOR_BUFFER_BIT);

    glActiveTexture(GL_TEXTURE0);
    glEnable(GL_TEXTURE_2D);
    glBindTexture(GL_TEXTURE_2D, tr[a]);

    glActiveTexture(GL_TEXTURE1);
    glDisable(GL_TEXTURE_2D);

    glUseProgram(shader_draw);
    glUniform1i(glGetUniformLocation(shader_draw, "tex0"), 0);
    glUniform1f(loc_colscheme, (float)colscheme);
    glUniform1f(loc_phase, (float)phase);

    glDisable(GL_DEPTH_TEST);

    glBegin(GL_QUADS);
    glMultiTexCoord2d(GL_TEXTURE0, fx, fy);
    glVertex2i(ox, oy);
    glMultiTexCoord2d(GL_TEXTURE0, fx + 1, fy);
    glVertex2i(ox + qx, oy);
    glMultiTexCoord2d(GL_TEXTURE0, fx + 1, fy + 1);
    glVertex2i(ox + qx, oy + qy);
    glMultiTexCoord2d(GL_TEXTURE0, fx, fy + 1);
    glVertex2i(ox, oy + qy);
    glEnd();

    glUseProgram(0);

  } else  // dims==3
  {
    glMatrixMode(GL_PROJECTION);
    glLoadIdentity();
    gluPerspective(30, (double)SX / SY, 0.1, 1000);
    glViewport(0, 0, SX, SY);

    glBindFramebuffer(GL_FRAMEBUFFER, 0);

    glActiveTexture(GL_TEXTURE0);
    glEnable(GL_TEXTURE_3D);
    glBindTexture(GL_TEXTURE_3D, tr[a]);
    glUseProgram(shader_draw);
    glUniform1i(glGetUniformLocation(shader_draw, "tex0"), 0);

    glUniform1f(glGetUniformLocation(shader_draw, "fx"), (float)fx);
    glUniform1f(glGetUniformLocation(shader_draw, "fy"), (float)fy);
    glUniform1f(glGetUniformLocation(shader_draw, "fz"), (float)fz);

    glUniform1f(glGetUniformLocation(shader_draw, "nx"), (float)NX);
    glUniform1f(glGetUniformLocation(shader_draw, "ny"), (float)NY);
    glUniform1f(glGetUniformLocation(shader_draw, "nz"), (float)NZ);

    glUniform1f(loc_colscheme, (float)colscheme);
    glUniform1f(loc_phase, (float)phase);
    glUniform1f(loc_visscheme, (float)visscheme);

    glTexParameterf(GL_TEXTURE_3D, GL_TEXTURE_MIN_FILTER, GL_LINEAR);
    glTexParameterf(GL_TEXTURE_3D, GL_TEXTURE_MAG_FILTER, GL_LINEAR);

    glEnable(GL_DEPTH_TEST);

    glClearColor(0.5, 0.5, 0.5, 1.0);
    glClearDepth(1.0);
    glClear(GL_COLOR_BUFFER_BIT | GL_DEPTH_BUFFER_BIT);

    glMatrixMode(GL_MODELVIEW);
    glLoadIdentity();

    glPushMatrix();
    glTranslated(0, 0, qq * 10);

    glRotated(-(wj + wy), 1, 0, 0);
    glRotated(-(wi + wx), 0, 1, 0);

    wi += dw;

    for (int s = 0; s < 6; s++) {
      glBegin(GL_QUADS);
      for (int c = 0; c < 4; c++) {
        glVertex3d(NX / 2 * cube[s][c][0], NY / 2 * cube[s][c][1],
                   NZ / 2 * cube[s][c][2]);
      }
      glEnd();
    }

    glPopMatrix();

    glUseProgram(0);
    glDisable(GL_DEPTH_TEST);
    glTexParameterf(GL_TEXTURE_3D, GL_TEXTURE_MIN_FILTER, GL_NEAREST);
    glTexParameterf(GL_TEXTURE_3D, GL_TEXTURE_MAG_FILTER, GL_NEAREST);
  }
}

// step types for kernel
//
double func_hard(double x, double a) {
  if (x >= a)
    return 1.0;
  else
    return 0.0;
}

double func_linear(double x, double a, double ea) {
  if (x < a - ea / 2.0)
    return 0.0;
  else if (x > a + ea / 2.0)
    return 1.0;
  else
    return (x - a) / ea + 0.5;
}

double func_hermite(double x, double a, double ea) {
  if (x < a - ea / 2.0)
    return 0.0;
  else if (x > a + ea / 2.0)
    return 1.0;
  else {
    double m = (x - (a - ea / 2.0)) / ea;
    return m * m * (3.0 - 2.0 * m);
  }
}

double func_sin(double x, double a, double ea) {
  if (x < a - ea / 2.0)
    return 0.0;
  else if (x > a + ea / 2.0)
    return 1.0;
  else
    return sin(PI / 2.0 * (x - a) / ea) * 0.5 + 0.5;
}

double func_smooth(double x, double a, double ea) {
  return 1.0 / (1.0 + exp(-(x - a) * 4.0 / ea));
}

double func_kernel(double x, double a, double ea) {
  // return func_hard   (x, a);
  return func_linear(x, a, ea);
  // return func_hermite (x, a, ea);
  // return func_sin     (x, a, ea);
  // return func_smooth  (x, a, ea);
}

// different kernel shapes
//
double pnorm(double x, double y, double z, double p) {
  return pow(pow(fabs(x), p) + pow(fabs(y), p) + pow(fabs(z), p), 1.0 / p);
}

// make the disk and ring kernel (real) buffers
//
void makekernel(int kr, int kd) {
  int ix, iy, iz, x, y, z;
  double l, n, m;
  float *ar, *ad;
  int Ra;
  double ri, bb;

  ad = (float *)calloc(NX * NY, sizeof(float));
  if (ad == 0) {
    fprintf(logfile, "ad failed\n");
    fflush(logfile);
    return;
  }

  ar = (float *)calloc(NX * NY, sizeof(float));
  if (ar == 0) {
    fprintf(logfile, "ar failed\n");
    fflush(logfile);
    return;
  }

  ri = ra / rr;
  bb = ra / rb;

  // Ra = (int)(ra+bb/2+1.0);
  Ra = (int)(ra * 2);

  kflr = 0.0;
  kfld = 0.0;

  for (iz = 0; iz < NZ; iz++) {
    memset(ad, 0, NX * NY * sizeof(float));
    memset(ar, 0, NX * NY * sizeof(float));
    if (dims > 2) {
      if (iz < NZ / 2)
        z = iz;
      else
        z = iz - NZ;
    } else {
      z = 0;
    }
    if (z >= -Ra && z <= Ra) {
      for (iy = 0; iy < NY; iy++) {
        if (dims > 1) {
          if (iy < NY / 2)
            y = iy;
          else
            y = iy - NY;
        } else {
          y = 0;
        }
        if (y >= -Ra && y <= Ra) {
          for (ix = 0; ix < NX; ix++) {
            if (ix < NX / 2)
              x = ix;
            else
              x = ix - NX;
            if (x >= -Ra && x <= Ra) {
              // l = pnorm (x,y,z,0.5);
              // l = pnorm (x,y,z,1);
              l = pnorm(x, y, z, 2);
              // l = pnorm (x,y,z,20);

              m = 1 - func_kernel(l, ri, bb);
              n = func_kernel(l, ri, bb) * (1 - func_kernel(l, ra, bb));

              /*
              // angle dependent kernel
              double w;
              w = atan2 (x, y);
              n *= 0.5*sin(6*w)+0.5;
              */

              /*
              // gauss kernel (profile is a gaussian instead of flat)
              const double mc = 2;
              const double nc = 4;

              double r;

              r = l/(ri/mc);
              m = exp(-r*r);

              r = l-(ra+ri)/2;
              r /= (ra-ri)/nc;
              n = exp(-r*r);
              */

              *(ad + iy * NX + ix) = (float)m;
              *(ar + iy * NX + ix) = (float)n;
              kflr += n;
              kfld += m;
            }  // if ix
          }    // for ix
        }      // if iy
      }        // for iy
    }          // if iz

    if (dims == 1) {
      glBindTexture(GL_TEXTURE_1D, tr[kd]);
      glTexSubImage1D(GL_TEXTURE_1D, 0, 0, NX, GL_RED, GL_FLOAT, ad);

      glBindTexture(GL_TEXTURE_1D, tr[kr]);
      glTexSubImage1D(GL_TEXTURE_1D, 0, 0, NX, GL_RED, GL_FLOAT, ar);
    } else if (dims == 2) {
      glBindTexture(GL_TEXTURE_2D, tr[kd]);
      glTexSubImage2D(GL_TEXTURE_2D, 0, 0, 0, NX, NY, GL_RED, GL_FLOAT, ad);

      glBindTexture(GL_TEXTURE_2D, tr[kr]);
      glTexSubImage2D(GL_TEXTURE_2D, 0, 0, 0, NX, NY, GL_RED, GL_FLOAT, ar);
    } else  // dims==3
    {
      glBindTexture(GL_TEXTURE_3D, tr[kd]);
      glTexSubImage3D(GL_TEXTURE_3D, 0, 0, 0, iz, NX, NY, 1, GL_RED, GL_FLOAT,
                      ad);

      glBindTexture(GL_TEXTURE_3D, tr[kr]);
      glTexSubImage3D(GL_TEXTURE_3D, 0, 0, 0, iz, NX, NY, 1, GL_RED, GL_FLOAT,
                      ar);
    }
  }  // for iz

  free(ar);
  free(ad);

  fprintf(
      logfile,
      "ra=%lf rr=%lf rb=%lf ri=%lf bb=%lf kflr=%lf kfld=%lf kflr/kfld=%lf\n",
      ra, rr, rb, ri, bb, kflr, kfld, kflr / kfld);
  fflush(logfile);
}

// make FFT plan buffers
//

int bitreverse(int x, int b) {
  int c, t;

  c = 0;
  for (t = 0; t < b; t++) {
    c = (c << 1) | ((x >> t) & 1);
  }
  return c;
}

void fft_planx(void) {
  float *p = (float *)calloc(4 * (NX / 2 + 1), sizeof(float));

  for (int s = 0; s <= 1; s++) {
    int si = s * 2 - 1;
    for (int eb = 0; eb <= BX - 1 + 1; eb++) {
      for (int x = 0; x < NX / 2 + 1; x++) {
        if (si == 1 && eb == 0) {
          *(p + 4 * x + 0) = (x + 0.5f) / (float)(NX / 2 + 1);
          *(p + 4 * x + 1) = (NX / 2 - x + 0.5f) / (float)(NX / 2 + 1);
          double w = si * PI * (x / (double)NX + 0.25);
          *(p + 4 * x + 2) = (float)cos(w);
          *(p + 4 * x + 3) = (float)sin(w);
        } else if (si == -1 && eb == BX) {
          if (x == 0 || x == NX / 2) {
            *(p + 4 * x + 0) = 0;
            *(p + 4 * x + 1) = 0;
          } else {
            *(p + 4 * x + 0) = (x) / (float)(NX / 2);
            *(p + 4 * x + 1) = (NX / 2 - x) / (float)(NX / 2);
          }
          double w = si * PI * (x / (double)NX + 0.25);
          *(p + 4 * x + 2) = (float)cos(w);
          *(p + 4 * x + 3) = (float)sin(w);
        } else if (x < NX / 2) {
          int l = 1 << eb;
          int j = x % l;
          double w = si * PI * j / l;
          if (j < l / 2) {
            if (eb == 1) {
              *(p + 4 * x + 0) = bitreverse(x, BX - 1) / (float)(NX / 2);
              *(p + 4 * x + 1) =
                  bitreverse(x + l / 2, BX - 1) / (float)(NX / 2);
            } else {
              *(p + 4 * x + 0) = (x) / (float)(NX / 2);
              *(p + 4 * x + 1) = (x + l / 2) / (float)(NX / 2);
            }
          } else {
            if (eb == 1) {
              *(p + 4 * x + 0) =
                  bitreverse(x - l / 2, BX - 1) / (float)(NX / 2);
              *(p + 4 * x + 1) = bitreverse(x, BX - 1) / (float)(NX / 2);
            } else {
              *(p + 4 * x + 0) = (x - l / 2) / (float)(NX / 2);
              *(p + 4 * x + 1) = (x) / (float)(NX / 2);
            }
          }
          *(p + 4 * x + 2) = (float)cos(w);
          *(p + 4 * x + 3) = (float)sin(w);
        } else {
          *(p + 4 * x + 0) = 0;
          *(p + 4 * x + 1) = 0;
          *(p + 4 * x + 2) = 0;
          *(p + 4 * x + 3) = 0;
        }
      }

      glBindTexture(GL_TEXTURE_1D, planx[eb][s]);
      glTexSubImage1D(GL_TEXTURE_1D, 0, 0, NX / 2 + 1, GL_RGBA, GL_FLOAT, p);
    }
  }

  free(p);
}

void fft_plany(void) {
  float *p = (float *)calloc(4 * NY, sizeof(float));

  for (int s = 0; s <= 1; s++) {
    int si = s * 2 - 1;
    for (int eb = 1; eb <= BY; eb++) {
      for (int x = 0; x < NY; x++) {
        int l = 1 << eb;
        int j = x % l;
        double w = si * PI * j / l;
        if (j < l / 2) {
          if (eb == 1) {
            *(p + 4 * x + 0) = bitreverse(x, BY) / (float)NY;
            *(p + 4 * x + 1) = bitreverse(x + l / 2, BY) / (float)NY;
          } else {
            *(p + 4 * x + 0) = (x) / (float)NY;
            *(p + 4 * x + 1) = (x + l / 2) / (float)NY;
          }
        } else {
          if (eb == 1) {
            *(p + 4 * x + 0) = bitreverse(x - l / 2, BY) / (float)NY;
            *(p + 4 * x + 1) = bitreverse(x, BY) / (float)NY;
          } else {
            *(p + 4 * x + 0) = (x - l / 2) / (float)NY;
            *(p + 4 * x + 1) = (x) / (float)NY;
          }
        }
        *(p + 4 * x + 2) = (float)cos(w);
        *(p + 4 * x + 3) = (float)sin(w);
      }

      glBindTexture(GL_TEXTURE_1D, plany[eb][s]);
      glTexSubImage1D(GL_TEXTURE_1D, 0, 0, NY, GL_RGBA, GL_FLOAT, p);
    }
  }

  free(p);
}

void fft_planz(void) {
  float *p = (float *)calloc(4 * NZ, sizeof(float));

  for (int s = 0; s <= 1; s++) {
    int si = s * 2 - 1;
    for (int eb = 1; eb <= BZ; eb++) {
      for (int x = 0; x < NZ; x++) {
        int l = 1 << eb;
        int j = x % l;
        double w = si * PI * j / l;
        if (j < l / 2) {
          if (eb == 1) {
            *(p + 4 * x + 0) = bitreverse(x, BZ) / (float)NZ;
            *(p + 4 * x + 1) = bitreverse(x + l / 2, BZ) / (float)NZ;
          } else {
            *(p + 4 * x + 0) = (x) / (float)NZ;
            *(p + 4 * x + 1) = (x + l / 2) / (float)NZ;
          }
        } else {
          if (eb == 1) {
            *(p + 4 * x + 0) = bitreverse(x - l / 2, BZ) / (float)NZ;
            *(p + 4 * x + 1) = bitreverse(x, BZ) / (float)NZ;
          } else {
            *(p + 4 * x + 0) = (x - l / 2) / (float)NZ;
            *(p + 4 * x + 1) = (x) / (float)NZ;
          }
        }
        *(p + 4 * x + 2) = (float)cos(w);
        *(p + 4 * x + 3) = (float)sin(w);
      }

      glBindTexture(GL_TEXTURE_1D, planz[eb][s]);
      glTexSubImage1D(GL_TEXTURE_1D, 0, 0, NZ, GL_RGBA, GL_FLOAT, p);
    }
  }

  free(p);
}

// copy a real buffer to a Fourier buffer
//
void copybufferrc(int vo, int na) {
  glMatrixMode(GL_PROJECTION);
  glLoadIdentity();
  glOrtho(0, NX / 2 + 1, 0, NY, -NZ, NZ);
  glViewport(0, 0, NX / 2 + 1, NY);

  glBindFramebuffer(GL_FRAMEBUFFER, fb[na]);
  glUseProgram(shader_copybufferrc);

  glActiveTexture(GL_TEXTURE0);
  glBindTexture(ttd, tr[vo]);
  glUniform1i(glGetUniformLocation(shader_copybufferrc, "tex0"), 0);

  glActiveTexture(GL_TEXTURE1);
  glBindTexture(ttd, tr[vo]);
  glUniform1i(glGetUniformLocation(shader_copybufferrc, "tex1"), 1);

  if (dims == 1) {
    glFramebufferTexture1D(GL_FRAMEBUFFER, GL_COLOR_ATTACHMENT0, GL_TEXTURE_1D,
                           tb[na], 0);
    glBegin(GL_LINES);
    glMultiTexCoord1d(GL_TEXTURE0, 0 - 1.0 / NX);
    glMultiTexCoord1d(GL_TEXTURE1, 0 - 0.0 / NX);
    glVertex2d(0, 0.5);
    glMultiTexCoord1d(GL_TEXTURE0, 1 - 1.0 / NX);
    glMultiTexCoord1d(GL_TEXTURE1, 1 - 0.0 / NX);
    glVertex2d(NX / 2, 0.5);
    glEnd();
  } else if (dims == 2) {
    glFramebufferTexture2D(GL_FRAMEBUFFER, GL_COLOR_ATTACHMENT0, GL_TEXTURE_2D,
                           tb[na], 0);
    glBegin(GL_QUADS);
    glMultiTexCoord2d(GL_TEXTURE0, 0 - 1.0 / NX, 0);
    glMultiTexCoord2d(GL_TEXTURE1, 0 - 0.0 / NX, 0);
    glVertex2i(0, 0);
    glMultiTexCoord2d(GL_TEXTURE0, 1 - 1.0 / NX, 0);
    glMultiTexCoord2d(GL_TEXTURE1, 1 - 0.0 / NX, 0);
    glVertex2i(NX / 2, 0);
    glMultiTexCoord2d(GL_TEXTURE0, 1 - 1.0 / NX, 1);
    glMultiTexCoord2d(GL_TEXTURE1, 1 - 0.0 / NX, 1);
    glVertex2i(NX / 2, NY);
    glMultiTexCoord2d(GL_TEXTURE0, 0 - 1.0 / NX, 1);
    glMultiTexCoord2d(GL_TEXTURE1, 0 - 0.0 / NX, 1);
    glVertex2i(0, NY);
    glEnd();
  } else  // dims==3
  {
    for (int t = 0; t < NZ; t++) {
      double l = (t + 0.5) / NZ;
      glFramebufferTexture3D(GL_FRAMEBUFFER, GL_COLOR_ATTACHMENT0,
                             GL_TEXTURE_3D, tb[na], 0, t);
      glBegin(GL_QUADS);
      glMultiTexCoord3d(GL_TEXTURE0, 0 - 1.0 / NX, 0, l);
      glMultiTexCoord3d(GL_TEXTURE1, 0 - 0.0 / NX, 0, l);
      glVertex3i(0, 0, t);
      glMultiTexCoord3d(GL_TEXTURE0, 1 - 1.0 / NX, 0, l);
      glMultiTexCoord3d(GL_TEXTURE1, 1 - 0.0 / NX, 0, l);
      glVertex3i(NX / 2, 0, t);
      glMultiTexCoord3d(GL_TEXTURE0, 1 - 1.0 / NX, 1, l);
      glMultiTexCoord3d(GL_TEXTURE1, 1 - 0.0 / NX, 1, l);
      glVertex3i(NX / 2, NY, t);
      glMultiTexCoord3d(GL_TEXTURE0, 0 - 1.0 / NX, 1, l);
      glMultiTexCoord3d(GL_TEXTURE1, 0 - 0.0 / NX, 1, l);
      glVertex3i(0, NY, t);
      glEnd();
    }
  }

  glUseProgram(0);
}

// copy a Fourier buffer to a real one
//
void copybuffercr(int vo, int na) {
  glMatrixMode(GL_PROJECTION);
  glLoadIdentity();
  glOrtho(0, NX, 0, NY, -NZ, NZ);
  glViewport(0, 0, NX, NY);

  glBindFramebuffer(GL_FRAMEBUFFER, fr[na]);
  glUseProgram(shader_copybuffercr);

  glActiveTexture(GL_TEXTURE0);
  glBindTexture(ttd, tb[vo]);
  glUniform1i(glGetUniformLocation(shader_copybuffercr, "tex0"), 0);

  glActiveTexture(GL_TEXTURE1);
  glBindTexture(ttd, tb[vo]);
  glUniform1i(glGetUniformLocation(shader_copybuffercr, "tex1"), 1);

  if (dims == 1) {
    glFramebufferTexture1D(GL_FRAMEBUFFER, GL_COLOR_ATTACHMENT0, GL_TEXTURE_1D,
                           tr[na], 0);
    glBegin(GL_LINES);
    glMultiTexCoord1d(GL_TEXTURE0, 0.0 / (NX / 2 + 1));
    glMultiTexCoord1d(GL_TEXTURE1, 0);
    glVertex2d(0, 0.5);
    glMultiTexCoord1d(GL_TEXTURE0, 1 - 1.0 / (NX / 2 + 1));
    glMultiTexCoord1d(GL_TEXTURE1, NX);
    glVertex2d(NX, 0.5);
    glEnd();
  } else if (dims == 2) {
    glFramebufferTexture2D(GL_FRAMEBUFFER, GL_COLOR_ATTACHMENT0, GL_TEXTURE_2D,
                           tr[na], 0);
    glBegin(GL_QUADS);
    glMultiTexCoord2d(GL_TEXTURE0, 0.0 / (NX / 2 + 1), 0);
    glMultiTexCoord2d(GL_TEXTURE1, 0, 0);
    glVertex2i(0, 0);
    glMultiTexCoord2d(GL_TEXTURE0, 1 - 1.0 / (NX / 2 + 1), 0);
    glMultiTexCoord2d(GL_TEXTURE1, NX, 0);
    glVertex2i(NX, 0);
    glMultiTexCoord2d(GL_TEXTURE0, 1 - 1.0 / (NX / 2 + 1), 1);
    glMultiTexCoord2d(GL_TEXTURE1, NX, 1);
    glVertex2i(NX, NY);
    glMultiTexCoord2d(GL_TEXTURE0, 0.0 / (NX / 2 + 1), 1);
    glMultiTexCoord2d(GL_TEXTURE1, 0, 1);
    glVertex2i(0, NY);
    glEnd();
  } else  // dims==3
  {
    for (int t = 0; t < NZ; t++) {
      double l = (t + 0.5) / NZ;
      glFramebufferTexture3D(GL_FRAMEBUFFER, GL_COLOR_ATTACHMENT0,
                             GL_TEXTURE_3D, tr[na], 0, t);
      glBegin(GL_QUADS);
      glMultiTexCoord3d(GL_TEXTURE0, 0.0 / (NX / 2 + 1), 0, l);
      glMultiTexCoord3d(GL_TEXTURE1, 0, 0, l);
      glVertex3i(0, 0, t);
      glMultiTexCoord3d(GL_TEXTURE0, 1 - 1.0 / (NX / 2 + 1), 0, l);
      glMultiTexCoord3d(GL_TEXTURE1, NX, 0, l);
      glVertex3i(NX, 0, t);
      glMultiTexCoord3d(GL_TEXTURE0, 1 - 1.0 / (NX / 2 + 1), 1, l);
      glMultiTexCoord3d(GL_TEXTURE1, NX, 1, l);
      glVertex3i(NX, NY, t);
      glMultiTexCoord3d(GL_TEXTURE0, 0.0 / (NX / 2 + 1), 1, l);
      glMultiTexCoord3d(GL_TEXTURE1, 0, 1, l);
      glVertex3i(0, NY, t);
      glEnd();
    }
  }

  glUseProgram(0);
}

// do an FFT stage
//
void fft_stage(int dim, int eb, int si, int fftc, int ffto) {
  glMatrixMode(GL_PROJECTION);
  glLoadIdentity();
  glOrtho(0, NX / 2 + 1, 0, NY, -NZ, NZ);
  glViewport(0, 0, NX / 2 + 1, NY);

  glBindFramebuffer(GL_FRAMEBUFFER, fb[ffto]);
  glUseProgram(shader_fft);
  glUniform1i(loc_dim, dim);

  int tang;
  double tangsc;
  if (dim == 1 && si == 1 && eb == 0) {
    tang = 1;
    tangsc = 0.5 * sqrt(2.0);
  } else if (dim == 1 && si == -1 && eb == BX) {
    tang = 1;
    tangsc = 0.5 / sqrt(2.0);
  } else {
    tang = 0;
    tangsc = 0.0;
  }
  glUniform1i(loc_tang, tang);
  glUniform1f(loc_tangsc, (float)tangsc);

  double gd;
  int gi;
  if (dim == 2 || dim == 3 || dim == 1 && si == -1 && eb == BX) {
    gd = 1.0;
    gi = NX / 2 + 1;
  } else {
    gd = 1.0 - 1.0 / (NX / 2 + 1);
    gi = NX / 2;
  }

  glActiveTexture(GL_TEXTURE0);
  glBindTexture(ttd, tb[fftc]);
  glUniform1i(glGetUniformLocation(shader_fft, "tex0"), 0);

  glActiveTexture(GL_TEXTURE1);
  if (dim == 1) glBindTexture(GL_TEXTURE_1D, planx[eb][(si + 1) / 2]);
  if (dim == 2) glBindTexture(GL_TEXTURE_1D, plany[eb][(si + 1) / 2]);
  if (dim == 3) glBindTexture(GL_TEXTURE_1D, planz[eb][(si + 1) / 2]);
  glUniform1i(glGetUniformLocation(shader_fft, "tex1"), 1);

  if (dims == 1) {
    glFramebufferTexture1D(GL_FRAMEBUFFER, GL_COLOR_ATTACHMENT0, GL_TEXTURE_1D,
                           tb[ffto], 0);
    glBegin(GL_LINES);
    glMultiTexCoord1d(GL_TEXTURE0, 0);
    glMultiTexCoord1d(GL_TEXTURE1, 0);
    glVertex2d(0, 0.5);
    glMultiTexCoord1d(GL_TEXTURE0, gd);
    glMultiTexCoord1d(GL_TEXTURE1, gd);
    glVertex2d(gi, 0.5);
    glEnd();
  } else if (dims == 2) {
    glFramebufferTexture2D(GL_FRAMEBUFFER, GL_COLOR_ATTACHMENT0, GL_TEXTURE_2D,
                           tb[ffto], 0);
    glBegin(GL_QUADS);
    glMultiTexCoord2d(GL_TEXTURE0, 0, 0);
    glMultiTexCoord2d(GL_TEXTURE1, 0, 0);
    glVertex2i(0, 0);
    glMultiTexCoord2d(GL_TEXTURE0, gd, 0);
    glMultiTexCoord2d(GL_TEXTURE1, gd, 0);
    glVertex2i(gi, 0);
    glMultiTexCoord2d(GL_TEXTURE0, gd, 1);
    glMultiTexCoord2d(GL_TEXTURE1, gd, 1);
    glVertex2i(gi, NY);
    glMultiTexCoord2d(GL_TEXTURE0, 0, 1);
    glMultiTexCoord2d(GL_TEXTURE1, 0, 1);
    glVertex2i(0, NY);
    glEnd();
  } else  // dims==3
  {
    for (int t = 0; t < NZ; t++) {
      double l = (t + 0.5) / NZ;
      glFramebufferTexture3D(GL_FRAMEBUFFER, GL_COLOR_ATTACHMENT0,
                             GL_TEXTURE_3D, tb[ffto], 0, t);
      glBegin(GL_QUADS);
      glMultiTexCoord3d(GL_TEXTURE0, 0, 0, l);
      glMultiTexCoord3d(GL_TEXTURE1, 0, 0, l);
      glVertex3i(0, 0, t);
      glMultiTexCoord3d(GL_TEXTURE0, gd, 0, l);
      glMultiTexCoord3d(GL_TEXTURE1, gd, 0, l);
      glVertex3i(gi, 0, t);
      glMultiTexCoord3d(GL_TEXTURE0, gd, 1, l);
      glMultiTexCoord3d(GL_TEXTURE1, gd, 1, l);
      glVertex3i(gi, NY, t);
      glMultiTexCoord3d(GL_TEXTURE0, 0, 1, l);
      glMultiTexCoord3d(GL_TEXTURE1, 0, 1, l);
      glVertex3i(0, NY, t);
      glEnd();
    }
  }

  glUseProgram(0);
}

// do the FFT on a buffer
//
void fft(int vo, int na, int si) {
  int t, s;
  int fftcur, fftoth;

  fftcur = FFT0;
  fftoth = FFT1;

  if (si == -1)  // real to Fourier
  {
    copybufferrc(vo, fftcur);

    for (t = 1; t <= BX - 1 + 1; t++) {
      if (dims == 1 && t == BX)
        fft_stage(1, t, si, fftcur, na);
      else
        fft_stage(1, t, si, fftcur, fftoth);
      s = fftcur;
      fftcur = fftoth;
      fftoth = s;
    }

    for (t = 1; t <= BY; t++) {
      if (dims == 2 && t == BY)
        fft_stage(2, t, si, fftcur, na);
      else
        fft_stage(2, t, si, fftcur, fftoth);
      s = fftcur;
      fftcur = fftoth;
      fftoth = s;
    }

    for (t = 1; t <= BZ; t++) {
      if (t == BZ)
        fft_stage(3, t, si, fftcur, na);
      else
        fft_stage(3, t, si, fftcur, fftoth);
      s = fftcur;
      fftcur = fftoth;
      fftoth = s;
    }
  } else  // si==1, Fourier to real
  {
    for (t = 1; t <= BZ; t++) {
      if (t == 1)
        fft_stage(3, t, si, vo, fftoth);
      else
        fft_stage(3, t, si, fftcur, fftoth);
      s = fftcur;
      fftcur = fftoth;
      fftoth = s;
    }

    for (t = 1; t <= BY; t++) {
      if (dims == 2 && t == 1)
        fft_stage(2, t, si, vo, fftoth);
      else
        fft_stage(2, t, si, fftcur, fftoth);
      s = fftcur;
      fftcur = fftoth;
      fftoth = s;
    }

    for (t = 0; t <= BX - 1; t++) {
      if (dims == 1 && t == 0)
        fft_stage(1, t, si, vo, fftoth);
      else
        fft_stage(1, t, si, fftcur, fftoth);
      s = fftcur;
      fftcur = fftoth;
      fftoth = s;
    }

    copybuffercr(fftcur, na);
  }
}

// multiply with kernel (Fourier buffers), scale
//
void kernelmul(int vo, int ke, int na, double sc) {
  glMatrixMode(GL_PROJECTION);
  glLoadIdentity();
  glOrtho(0, NX / 2 + 1, 0, NY, -NZ, NZ);
  glViewport(0, 0, NX / 2 + 1, NY);

  glBindFramebuffer(GL_FRAMEBUFFER, fb[na]);
  glUseProgram(shader_kernelmul);
  glUniform1f(loc_sc, (float)sc);

  glActiveTexture(GL_TEXTURE0);
  glBindTexture(ttd, tb[vo]);
  glUniform1i(glGetUniformLocation(shader_kernelmul, "tex0"), 0);

  glActiveTexture(GL_TEXTURE1);
  glBindTexture(ttd, tb[ke]);
  glUniform1i(glGetUniformLocation(shader_kernelmul, "tex1"), 1);

  if (dims == 1) {
    glFramebufferTexture1D(GL_FRAMEBUFFER, GL_COLOR_ATTACHMENT0, GL_TEXTURE_1D,
                           tb[na], 0);
    glBegin(GL_LINES);
    glMultiTexCoord1d(GL_TEXTURE0, 0);
    glMultiTexCoord1d(GL_TEXTURE1, 0);
    glVertex2d(0, 0.5);
    glMultiTexCoord1d(GL_TEXTURE0, 1);
    glMultiTexCoord1d(GL_TEXTURE1, 1);
    glVertex2d(NX / 2 + 1, 0.5);
    glEnd();
  } else if (dims == 2) {
    glFramebufferTexture2D(GL_FRAMEBUFFER, GL_COLOR_ATTACHMENT0, GL_TEXTURE_2D,
                           tb[na], 0);
    glBegin(GL_QUADS);
    glMultiTexCoord2d(GL_TEXTURE0, 0, 0);
    glMultiTexCoord2d(GL_TEXTURE1, 0, 0);
    glVertex2i(0, 0);
    glMultiTexCoord2d(GL_TEXTURE0, 1, 0);
    glMultiTexCoord2d(GL_TEXTURE1, 1, 0);
    glVertex2i(NX / 2 + 1, 0);
    glMultiTexCoord2d(GL_TEXTURE0, 1, 1);
    glMultiTexCoord2d(GL_TEXTURE1, 1, 1);
    glVertex2i(NX / 2 + 1, NY);
    glMultiTexCoord2d(GL_TEXTURE0, 0, 1);
    glMultiTexCoord2d(GL_TEXTURE1, 0, 1);
    glVertex2i(0, NY);
    glEnd();
  } else  // dims==3
  {
    for (int t = 0; t < NZ; t++) {
      double l = (t + 0.5) / NZ;
      glFramebufferTexture3D(GL_FRAMEBUFFER, GL_COLOR_ATTACHMENT0,
                             GL_TEXTURE_3D, tb[na], 0, t);
      glBegin(GL_QUADS);
      glMultiTexCoord3d(GL_TEXTURE0, 0, 0, l);
      glMultiTexCoord3d(GL_TEXTURE1, 0, 0, l);
      glVertex3i(0, 0, t);
      glMultiTexCoord3d(GL_TEXTURE0, 1, 0, l);
      glMultiTexCoord3d(GL_TEXTURE1, 1, 0, l);
      glVertex3i(NX / 2 + 1, 0, t);
      glMultiTexCoord3d(GL_TEXTURE0, 1, 1, l);
      glMultiTexCoord3d(GL_TEXTURE1, 1, 1, l);
      glVertex3i(NX / 2 + 1, NY, t);
      glMultiTexCoord3d(GL_TEXTURE0, 0, 1, l);
      glMultiTexCoord3d(GL_TEXTURE1, 0, 1, l);
      glVertex3i(0, NY, t);
      glEnd();
    }
  }

  glUseProgram(0);
}

// apply the snm function (real buffers)
//
void snm(int an, int am, int na) {
  glMatrixMode(GL_PROJECTION);
  glLoadIdentity();
  glOrtho(0, NX, 0, NY, -NZ, NZ);
  glViewport(0, 0, NX, NY);

  glBindFramebuffer(GL_FRAMEBUFFER, fr[na]);
  glUseProgram(shader_snm);
  glUniform1f(loc_mode, (float)mode);
  glUniform1f(loc_dt, (float)dt);
  glUniform1f(loc_b1, (float)b1);
  glUniform1f(loc_b2, (float)b2);
  glUniform1f(loc_d1, (float)d1);
  glUniform1f(loc_d2, (float)d2);
  glUniform1f(loc_sigmode, (float)sigmode);
  glUniform1f(loc_sigtype, (float)sigtype);
  glUniform1f(loc_mixtype, (float)mixtype);
  glUniform1f(loc_sn, (float)sn);
  glUniform1f(loc_sm, (float)sm);

  glActiveTexture(GL_TEXTURE0);
  glBindTexture(ttd, tr[an]);
  glUniform1i(glGetUniformLocation(shader_snm, "tex0"), 0);

  glActiveTexture(GL_TEXTURE1);
  glBindTexture(ttd, tr[am]);
  glUniform1i(glGetUniformLocation(shader_snm, "tex1"), 1);

  glActiveTexture(GL_TEXTURE2);
  glBindTexture(ttd, tr[na]);
  glUniform1i(glGetUniformLocation(shader_snm, "tex2"), 2);

  if (dims == 1) {
    glFramebufferTexture1D(GL_FRAMEBUFFER, GL_COLOR_ATTACHMENT0, GL_TEXTURE_1D,
                           tr[na], 0);
    glBegin(GL_LINES);
    glMultiTexCoord1d(GL_TEXTURE0, 0);
    glMultiTexCoord1d(GL_TEXTURE1, 0);
    glMultiTexCoord2d(GL_TEXTURE2, 0, 0);
    glVertex2d(0, 0.5);
    glMultiTexCoord1d(GL_TEXTURE0, 1);
    glMultiTexCoord1d(GL_TEXTURE1, 1);
    glMultiTexCoord2d(GL_TEXTURE2, 1, 0);
    glVertex2d(NX, 0.5);
    glEnd();
  } else if (dims == 2) {
    glFramebufferTexture2D(GL_FRAMEBUFFER, GL_COLOR_ATTACHMENT0, GL_TEXTURE_2D,
                           tr[na], 0);
    glBegin(GL_QUADS);
    glMultiTexCoord2d(GL_TEXTURE0, 0, 0);
    glMultiTexCoord2d(GL_TEXTURE1, 0, 0);
    glMultiTexCoord2d(GL_TEXTURE2, 0, 0);
    glVertex2i(0, 0);
    glMultiTexCoord2d(GL_TEXTURE0, 1, 0);
    glMultiTexCoord2d(GL_TEXTURE1, 1, 0);
    glMultiTexCoord2d(GL_TEXTURE2, 1, 0);
    glVertex2i(NX, 0);
    glMultiTexCoord2d(GL_TEXTURE0, 1, 1);
    glMultiTexCoord2d(GL_TEXTURE1, 1, 1);
    glMultiTexCoord2d(GL_TEXTURE2, 1, 1);
    glVertex2i(NX, NY);
    glMultiTexCoord2d(GL_TEXTURE0, 0, 1);
    glMultiTexCoord2d(GL_TEXTURE1, 0, 1);
    glMultiTexCoord2d(GL_TEXTURE2, 0, 1);
    glVertex2i(0, NY);
    glEnd();
  } else  // dims==3
  {
    for (int t = 0; t < NZ; t++) {
      double l = (t + 0.5) / NZ;
      glFramebufferTexture3D(GL_FRAMEBUFFER, GL_COLOR_ATTACHMENT0,
                             GL_TEXTURE_3D, tr[na], 0, t);
      glBegin(GL_QUADS);
      glMultiTexCoord3d(GL_TEXTURE0, 0, 0, l);
      glMultiTexCoord3d(GL_TEXTURE1, 0, 0, l);
      glMultiTexCoord3d(GL_TEXTURE2, 0, 0, l);
      glVertex3i(0, 0, t);
      glMultiTexCoord3d(GL_TEXTURE0, 1, 0, l);
      glMultiTexCoord3d(GL_TEXTURE1, 1, 0, l);
      glMultiTexCoord3d(GL_TEXTURE2, 1, 0, l);
      glVertex3i(NX, 0, t);
      glMultiTexCoord3d(GL_TEXTURE0, 1, 1, l);
      glMultiTexCoord3d(GL_TEXTURE1, 1, 1, l);
      glMultiTexCoord3d(GL_TEXTURE2, 1, 1, l);
      glVertex3i(NX, NY, t);
      glMultiTexCoord3d(GL_TEXTURE0, 0, 1, l);
      glMultiTexCoord3d(GL_TEXTURE1, 0, 1, l);
      glMultiTexCoord3d(GL_TEXTURE2, 0, 1, l);
      glVertex3i(0, NY, t);
      glEnd();
    }
  }

  glUseProgram(0);
}

// initialize an and am with 0 to 1 gradient for drawing of snm (2D only)
//
void initan(int a) {
  float *buf = (float *)calloc(NX * NY, sizeof(float));

  int x, y;

  for (y = 0; y < NY; y++)
    for (x = 0; x < NX; x++) {
      buf[y * NX + x] = (float)x / NX;
    }

  glBindTexture(GL_TEXTURE_2D, tr[a]);
  glTexImage2D(GL_TEXTURE_2D, 0, GL_R32F, NX, NY, 0, GL_RED, GL_FLOAT, buf);

  free(buf);
}

void initam(int a) {
  float *buf = (float *)calloc(NX * NY, sizeof(float));

  int x, y;

  for (y = 0; y < NY; y++)
    for (x = 0; x < NX; x++) {
      buf[y * NX + x] = (float)y / NY;
    }

  glBindTexture(GL_TEXTURE_2D, tr[a]);
  glTexImage2D(GL_TEXTURE_2D, 0, GL_R32F, NX, NY, 0, GL_RED, GL_FLOAT, buf);

  free(buf);
}

// draw snm function into buffer (2D only)
//
void makesnm(int an, int am, int asnm) {
  initan(an);
  initam(am);
  snm(an, am, asnm);
}

// window proc
//
int doevents(void) {
  int sym, wParam;
  int xp, yp;  // for mouse dragging
  static int lbutton = 0, rbutton = 0;
  static int oxp, oyp;
  static int oox, ooy;

  SDL_Event event;

  while (SDL_PollEvent(&event)) {
    switch (event.type) {
      case SDL_QUIT:
        return 1;
        break;

      case SDL_KEYDOWN:
        sym = event.key.keysym.sym;
        wParam = event.key.keysym.unicode;

        if (wParam == 27) return 1;

        if (wParam == 'b' || wParam == 'n' || wParam == ' ') inita(AA);

        if (wParam == 'p') pause ^= 1;

        if (wParam == '(' || wParam == ')') {
          if (wParam == '(') curparas--;
          if (curparas < 0) curparas = 0;
          if (wParam == ')') curparas++;
          if (curparas >= nparas) curparas = nparas - 1;
          setparas(curparas);
          neuedim = true;
          delShaders();
          delete_buffers();
          savedispcnt = 5.0;
          sprintf(dispmessage, " %d %s ", curparas, paralist[curparas].desc);
        }

        if (wParam == 'c') {
          anz++;
          if (anz > 4) anz = 1;
        }
        if (wParam == 'v') timing ^= 1;

        if (wParam == 'C') {
          visscheme++;
          if (visscheme > 6) visscheme = 0;
        }
        if (wParam == 'V') {
          visscheme--;
          if (visscheme < 0) visscheme = 6;
        }

        if (wParam == 'x') {
          colscheme++;
          if (colscheme > 7) colscheme = 1;
        }
        if (wParam == 'y') {
          colscheme--;
          if (colscheme < 1) colscheme = 7;
        }

        if (wParam == 'X') phase += 0.01;
        if (wParam == 'Y') phase -= 0.01;

        if (wParam == 't') {
          sigmode++;
          if (sigmode > 4) sigmode = 1;
        }
        if (wParam == 'g') {
          sigmode--;
          if (sigmode < 1) sigmode = 4;
        }

        if (wParam == 'z') {
          sigtype++;
          if (sigtype > 9) sigtype = 0;
        }
        if (wParam == 'h') {
          sigtype--;
          if (sigtype < 0) sigtype = 9;
        }

        if (wParam == 'u') {
          mixtype++;
          if (mixtype > 7) mixtype = 0;
        }
        if (wParam == 'j') {
          mixtype--;
          if (mixtype < 0) mixtype = 7;
        }

        if (wParam == '0') mode = 0;
        if (wParam == '1') mode = 1;
        if (wParam == '2') mode = 2;
        if (wParam == '3') mode = 3;
        if (wParam == '4') mode = 4;

        if (wParam == 'T') {
          ra += 0.1;
          makekernel(KR, KD);
          fft(KR, KRF, -1);
          fft(KD, KDF, -1);
        }
        if (wParam == 'G') {
          ra -= 0.1;
          if (ra < 1.0) ra = 1.0;
          makekernel(KR, KD);
          fft(KR, KRF, -1);
          fft(KD, KDF, -1);
        }

        if (wParam == 'Z') dt += 0.001;
        if (wParam == 'H') {
          dt -= 0.001;
          if (dt < 0.001) dt = 0.001;
        }

        if (dims == 1) {
          if (wParam == '5') {
            delete_buffers();
            NX = 512;
            BX = 9;
            neu = true;
          }
          if (wParam == '6') {
            delete_buffers();
            NX = 1024;
            BX = 10;
            neu = true;
          }
          if (wParam == '7') {
            delete_buffers();
            NX = 2048;
            BX = 11;
            neu = true;
          }
          if (wParam == '8') {
            delete_buffers();
            NX = 4096;
            BX = 12;
            neu = true;
          }
          if (wParam == '9') {
            delete_buffers();
            NX = 8192;
            BX = 13;
            neu = true;
          }
        } else if (dims == 2) {
          if (wParam == '5') {
            delete_buffers();
            NX = 128;
            NY = 128;
            BX = 7;
            BY = 7;
            neu = true;
          }
          if (wParam == '6') {
            delete_buffers();
            NX = 256;
            NY = 256;
            BX = 8;
            BY = 8;
            neu = true;
          }
          if (wParam == '7') {
            delete_buffers();
            NX = 512;
            NY = 512;
            BX = 9;
            BY = 9;
            neu = true;
          }
          if (wParam == '8') {
            delete_buffers();
            NX = 1024;
            NY = 1024;
            BX = 10;
            BY = 10;
            neu = true;
          }
          if (wParam == '9') {
            delete_buffers();
            NX = 2048;
            NY = 2048;
            BX = 11;
            BY = 11;
            neu = true;
          }
        } else  // dims==3
        {
          if (wParam == '5') {
            delete_buffers();
            NX = 32;
            NY = 32;
            NZ = 32;
            BX = 5;
            BY = 5;
            BZ = 5;
            neu = true;
          }
          if (wParam == '6') {
            delete_buffers();
            NX = 64;
            NY = 64;
            NZ = 64;
            BX = 6;
            BY = 6;
            BZ = 6;
            neu = true;
          }
          if (wParam == '7') {
            delete_buffers();
            NX = 128;
            NY = 128;
            NZ = 128;
            BX = 7;
            BY = 7;
            BZ = 7;
            neu = true;
          }
          if (wParam == '8') {
            delete_buffers();
            NX = 256;
            NY = 256;
            NZ = 256;
            BX = 8;
            BY = 8;
            BZ = 8;
            neu = true;
          }
          if (wParam == '9') {
            delete_buffers();
            NX = 512;
            NY = 512;
            NZ = 512;
            BX = 9;
            BY = 9;
            BZ = 9;
            neu = true;
          }
        }

        if (wParam == 'q') b1 += 0.001;
        if (wParam == 'a') b1 -= 0.001;
        if (wParam == 'w') b2 += 0.001;
        if (wParam == 's') b2 -= 0.001;

        if (wParam == 'e') d1 += 0.001;
        if (wParam == 'd') d1 -= 0.001;
        if (wParam == 'r') d2 += 0.001;
        if (wParam == 'f') d2 -= 0.001;

        if (wParam == 'i') sn += 0.001;
        if (wParam == 'k') sn -= 0.001;

        if (wParam == 'o') sm += 0.001;
        if (wParam == 'l') sm -= 0.001;

        if (wParam == 'Q') b1 += 0.01;
        if (wParam == 'A') b1 -= 0.01;
        if (wParam == 'W') b2 += 0.01;
        if (wParam == 'S') b2 -= 0.01;

        if (wParam == 'E') d1 += 0.01;
        if (wParam == 'D') d1 -= 0.01;
        if (wParam == 'R') d2 += 0.01;
        if (wParam == 'F') d2 -= 0.01;

        if (wParam == 'I') sn += 0.01;
        if (wParam == 'K') sn -= 0.01;

        if (wParam == 'O') sm += 0.01;
        if (wParam == 'L') sm -= 0.01;

        if (wParam == 'm') {
          write_config();
          savedispcnt = 1.0;
          sprintf(dispmessage, " saved. ");
        }

        if (wParam == '-') {
          if (dims == 2) mysavepic();
        }

        if (wParam == '.') {
          if (maximized) {
            SX = oldSX;
            SY = oldSY;
            SDL_SetVideoMode(SX, SY, 32,
                             SDL_HWSURFACE | SDL_GL_DOUBLEBUFFER | SDL_OPENGL);
            maximized = 0;
          } else {
            oldSX = SX;
            oldSY = SY;
            SX = DX;
            SY = DY;
            SDL_SetVideoMode(SX, SY, 32,
                             SDL_FULLSCREEN | SDL_HWSURFACE |
                                 SDL_GL_DOUBLEBUFFER | SDL_OPENGL);
            maximized = 1;
          }
          neuedim = true;
        }

        if (wParam == ',') {
          ox = 0;
          oy = 3 * 20;
          SX = 512;
          SY = 512 + 3 * 20;
          oldSX = SX;
          oldSY = SY;
          SDL_SetVideoMode(SX, SY, 32,
                           SDL_HWSURFACE | SDL_GL_DOUBLEBUFFER | SDL_OPENGL);
          maximized = 0;
          neuedim = true;
        }

        if (sym == SDLK_F1) dw += 0.05;  // 3D cube rotation speed
        if (sym == SDLK_F2) dw -= 0.05;

        if (sym == SDLK_F3)
          dphase += 0.000001 * pow(10, dims);  // color phase speed
        if (sym == SDLK_F4) dphase -= 0.000001 * pow(10, dims);

        if (sym == SDLK_F5) {
          dims = 1;
          neuedim = true;
        }
        if (sym == SDLK_F6) {
          dims = 2;
          neuedim = true;
        }
        if (sym == SDLK_F7) {
          dims = 3;
          neuedim = true;
        }

        if (sym == SDLK_LEFT) fx -= 0.01;  // cursor keys
        if (sym == SDLK_RIGHT) fx += 0.01;
        if (sym == SDLK_DOWN) fy -= 0.01;
        if (sym == SDLK_UP) fy += 0.01;
        if (sym == SDLK_PAGEUP) fz -= 0.01;
        if (sym == SDLK_PAGEDOWN) fz += 0.01;

        break;

      case SDL_MOUSEBUTTONDOWN:
        if (event.button.button == SDL_BUTTON_LEFT) {
          oxp = event.button.x;
          oyp = event.button.y;
          oox = ox;
          ooy = oy;
          lbutton = 1;
        }
        break;

      case SDL_MOUSEBUTTONUP:
        if (event.button.button == SDL_BUTTON_LEFT) {
          lbutton = 0;
          if (dims == 3) {
            wi = wi + wx;
            wx = 0;
            wj = wj + wy;
            wy = 0;
          }
        }
        if (event.button.button == SDL_BUTTON_WHEELUP) {
          qq++;
          qx = (int)pow(2, qq / 16.0);
          qy = (int)pow(2, qq / 16.0);
        }
        if (event.button.button == SDL_BUTTON_WHEELDOWN) {
          qq--;
          qx = (int)pow(2, qq / 16.0);
          qy = (int)pow(2, qq / 16.0);
        }
        break;

      case SDL_MOUSEMOTION:
        xp = event.motion.x;
        yp = event.motion.y;
        if (lbutton) {
          if (dims == 2) {
            ox = oox + (xp - oxp);
            oy = ooy - (yp - oyp);
          } else if (dims == 3) {
            wy = oyp - yp;
            wx = oxp - xp;
          }
        }
        break;
    }
  }
  return 0;
}

// draw a text on screen with the pre-rendered character textures
//
void drawtext(double ln, char *text) {
  int t, l, c, w, h, x;

  l = strlen(text);
  x = 0;

  glActiveTexture(GL_TEXTURE0);
  glEnable(GL_TEXTURE_2D);

  for (t = 0; t < l; t++) {
    c = text[t];
    w = fontw[c];
    h = fonth[c];

    glBindTexture(GL_TEXTURE_2D, fonttex[c]);
    glBegin(GL_QUADS);
    glTexCoord2d(0, 1);
    glVertex2d(x, ln * 20);
    glTexCoord2d(1, 1);
    glVertex2d(x + w, ln * 20);
    glTexCoord2d(1, 0);
    glVertex2d(x + w, ln * 20 + h);
    glTexCoord2d(0, 0);
    glVertex2d(x, ln * 20 + h);
    glEnd();
    x += w;
  }
}

// make a small texture for every single character
//
void makefonttextures(TTF_Font *font) {
  int t;
  unsigned char txt[8];

  glGenTextures(256, &fonttex[0]);
  for (t = 0; t < 256; t++) {
    fontw[t] = 0;
    fonth[t] = 0;
  }

  for (t = 32; t < 128; t++) {
    SDL_Color fg = {255, 255, 255};
    SDL_Color bg = {0, 0, 0};

    txt[0] = t;
    txt[1] = 0;
    SDL_Surface *message = TTF_RenderText_Shaded(font, (char *)&txt[0], fg, bg);

    SDL_PixelFormat fmt;
    fmt.palette = 0;
    fmt.BitsPerPixel = 24;
    fmt.BytesPerPixel = 3;
    fmt.Rmask = 0xff0000;
    fmt.Gmask = 0x00ff00;
    fmt.Bmask = 0x0000ff;
    fmt.Amask = 0;

    SDL_Surface *surf = SDL_ConvertSurface(message, &fmt, 0);

    glActiveTexture(GL_TEXTURE0);
    glEnable(GL_TEXTURE_2D);
    glBindTexture(GL_TEXTURE_2D, fonttex[t]);

    glTexParameterf(GL_TEXTURE_2D, GL_TEXTURE_MIN_FILTER, GL_LINEAR);
    glTexParameterf(GL_TEXTURE_2D, GL_TEXTURE_MAG_FILTER, GL_LINEAR);

    glTexImage2D(GL_TEXTURE_2D, 0, GL_RGB, surf->w, surf->h, 0, GL_BGR,
                 GL_UNSIGNED_BYTE, surf->pixels);
    fontw[t] = surf->w;
    fonth[t] = surf->h;

    SDL_FreeSurface(surf);
    SDL_FreeSurface(message);
  }
}

// Main
//
int main(int argc, char *argv[]) {
  TTF_Font *font;
  SDL_Surface *screen;

  screen = 0;
  font = 0;
  logfile = 0;
  ttd = 0;

  srand((unsigned)time(0));

  logfile = fopen("SmoothLifeLog.txt", "wt");
  if (logfile == 0) goto ende;
  fprintf(logfile, "starting SmoothLife\n");
  fflush(logfile);

  if (!read_config()) {
    fprintf(logfile, "couldn't read config file\n");
    fflush(logfile);
    goto ende;
  }

  SDL_Init(SDL_INIT_EVERYTHING);

  TTF_Init();
  // font = TTF_OpenFont ("Courier.ttf", 16);
  // font = TTF_OpenFont ("Arial.ttf", 15);
  font = TTF_OpenFont(
      "/home/louis/.local/share/unity3d/Packages/node_modules/"
      "unity-editor-home/dist/assets/font/Roboto/Roboto-Light.ttf",
      15);
  if (font == 0) {
    fprintf(logfile, "couldn't load font\n");
    fflush(logfile);
    goto ende;
  }
  TTF_SetFontStyle(font, TTF_STYLE_BOLD);

  SDL_EnableUNICODE(1);
  SDL_EnableKeyRepeat(180, 14);

  SDL_VideoInfo *vinf;
  vinf = (SDL_VideoInfo *)SDL_GetVideoInfo();
  DX = vinf->current_w;
  DY = vinf->current_h;
  fprintf(logfile, "desktop resolution %d %d\n", DX, DY);
  fflush(logfile);

  SDL_GL_SetAttribute(SDL_GL_MULTISAMPLEBUFFERS, 1);
  SDL_GL_SetAttribute(SDL_GL_MULTISAMPLESAMPLES, 8);
  maximized = 0;
  // SX = 800; SY = 600;
  SX = 340;
  SY = 200;
  oldSX = SX;
  oldSY = SY;
  screen = SDL_SetVideoMode(SX, SY, 32,
                            SDL_HWSURFACE | SDL_GL_DOUBLEBUFFER | SDL_OPENGL);
  SDL_WM_SetCaption(prgname, 0);

  const GLubyte *str;
  str = glGetString(GL_VENDOR);
  fprintf(logfile, "glvendor %s\n", str);
  fflush(logfile);
  str = glGetString(GL_RENDERER);
  fprintf(logfile, "glrenderer %s\n", str);
  fflush(logfile);
  str = glGetString(GL_VERSION);
  fprintf(logfile, "glversion %s\n", str);
  fflush(logfile);
  str = glGetString(GL_SHADING_LANGUAGE_VERSION);
  fprintf(logfile, "glslversion %s\n", str);
  fflush(logfile);

  // wglSwapIntervalEXT (0);		// switch off vsync (windows only, comment out
  // else)

  ypos = 0;
  savedispcnt = 0.0;
  timing = 1;
  colscheme = 2;
  visscheme = 2;
  anz = 1;
  pause = 0;
  ox = 10;
  oy = 70;
  phase = 0.0;
  dphase = 0.0;
  fx = 0;
  fy = 0;
  fz = 0;
  wi = 0;
  wj = 0;
  wx = 0;
  wy = 0;
  dw = 0.0;
  curparas = 0;
  setparas(curparas);

// dimension has changed, keys '(' and ')', or f5/f6/f7
neuedim:

  makefonttextures(font);

  if (dims == 1) ttd = GL_TEXTURE_1D;
  if (dims == 2) ttd = GL_TEXTURE_2D;
  if (dims == 3) ttd = GL_TEXTURE_3D;

  if (setShaders(dims, (char *)"copybufferrc", shader_copybufferrc)) goto ende;
  if (setShaders(dims, (char *)"copybuffercr", shader_copybuffercr)) goto ende;
  if (setShaders(dims, (char *)"fft", shader_fft)) goto ende;
  if (setShaders(dims, (char *)"kernelmul", shader_kernelmul)) goto ende;
  if (setShaders(dims, (char *)"snm", shader_snm)) goto ende;
  if (setShaders(dims, (char *)"draw", shader_draw)) goto ende;

  loc_dim = glGetUniformLocation(shader_fft, "dim");
  loc_tang = glGetUniformLocation(shader_fft, "tang");
  loc_tangsc = glGetUniformLocation(shader_fft, "tangsc");

  loc_sc = glGetUniformLocation(shader_kernelmul, "sc");

  loc_mode = glGetUniformLocation(shader_snm, "mode");
  loc_dt = glGetUniformLocation(shader_snm, "dt");
  loc_b1 = glGetUniformLocation(shader_snm, "b1");
  loc_b2 = glGetUniformLocation(shader_snm, "b2");
  loc_d1 = glGetUniformLocation(shader_snm, "d1");
  loc_d2 = glGetUniformLocation(shader_snm, "d2");
  loc_sigmode = glGetUniformLocation(shader_snm, "sigmode");
  loc_sigtype = glGetUniformLocation(shader_snm, "sigtype");
  loc_mixtype = glGetUniformLocation(shader_snm, "mixtype");
  loc_sn = glGetUniformLocation(shader_snm, "sn");
  loc_sm = glGetUniformLocation(shader_snm, "sm");

  loc_colscheme = glGetUniformLocation(shader_draw, "colscheme");
  loc_phase = glGetUniformLocation(shader_draw, "phase");
  loc_visscheme = glGetUniformLocation(shader_draw, "visscheme");

  glClampColor(GL_CLAMP_FRAGMENT_COLOR, GL_FALSE);
  glClampColor(GL_CLAMP_VERTEX_COLOR, GL_FALSE);
  glClampColor(GL_CLAMP_READ_COLOR, GL_FALSE);

  glEnable(GL_DEPTH_TEST);
  glEnable(GL_CULL_FACE);

  if (dims == 1) {
    NX = 1024;
    BX = 10;  // 262144, 18
    NY = 1;
    BY = 0;
    NZ = 1;
    BZ = 0;
  } else if (dims == 2) {
    NX = 512;
    BX = 9;
    NY = 512;
    BY = 9;
    NZ = 1;
    BZ = 0;
  } else  // dims==3
  {
    NX = 64;
    BX = 6;
    NY = 64;
    BY = 6;
    NZ = 64;
    BZ = 6;
  }

// buffer size has changed (keys 5,6,7,8,9)
nochmal:

  qx = NX;
  qy = NY;
  if (dims == 1) qq = BX * 16;
  if (dims == 2) qq = BX * 16;
  if (dims == 3) qq = -NX * 3 / 10;

  if (!create_buffers()) goto ende;

  fft_planx();
  if (dims > 1) fft_plany();
  if (dims > 2) fft_planz();
  makekernel(KR, KD);
  fft(KR, KRF, -1);
  fft(KD, KDF, -1);

  inita(AA);

  neu = false;
  neuedim = false;

  for (;;)  // main loop
  {
    if (doevents()) goto ende;

    if (neu) goto nochmal;      // new buffer size
    if (neuedim) goto neuedim;  // new parameter set

    if (timing) {
      glFlush();
      glFinish();
      tim = SDL_GetTicks();
    }

    if (anz == 1)  // draw buffer and do a time step
    {
      drawa(AA);
      if (!pause) {
        fft(AA, AF, -1);
        kernelmul(AF, KRF, ANF, sqrt(NX * NY * NZ) / kflr);
        kernelmul(AF, KDF, AMF, sqrt(NX * NY * NZ) / kfld);
        fft(ANF, AN, 1);
        fft(AMF, AM, 1);
        snm(AN, AM, AA);
        phase += dphase;
        ypos++;
        if (ypos >= SY) ypos = 0;
      }
    } else if (anz == 2)  // draw snm function
    {
      if (dims == 2) makesnm(AN, AM, AA);
      drawa(AA);
    } else if (anz == 3)  // draw disk kernel
    {
      drawa(KD);
    } else if (anz == 4)  // draw ring kernel
    {
      drawa(KR);
    }

    if (timing)  // show info
    {
      glFlush();
      glFinish();
      tima = SDL_GetTicks();

      glMatrixMode(GL_PROJECTION);
      glLoadIdentity();
      glOrtho(0, SX, 0, SY, -1, 1);
      glViewport(0, 0, SX, SY);
      glMatrixMode(GL_MODELVIEW);
      glLoadIdentity();

      glBindFramebuffer(GL_FRAMEBUFFER, 0);
      glBindTexture(GL_TEXTURE_1D, 0);
      glBindTexture(GL_TEXTURE_2D, 0);
      glBindTexture(GL_TEXTURE_3D, 0);

      glActiveTexture(GL_TEXTURE0);
      glDisable(GL_TEXTURE_1D);
      glDisable(GL_TEXTURE_2D);
      glDisable(GL_TEXTURE_3D);
      glActiveTexture(GL_TEXTURE1);
      glDisable(GL_TEXTURE_1D);
      glDisable(GL_TEXTURE_2D);
      glDisable(GL_TEXTURE_3D);
      glActiveTexture(GL_TEXTURE2);
      glDisable(GL_TEXTURE_1D);
      glDisable(GL_TEXTURE_2D);
      glDisable(GL_TEXTURE_3D);
      glActiveTexture(GL_TEXTURE3);
      glDisable(GL_TEXTURE_1D);
      glDisable(GL_TEXTURE_2D);
      glDisable(GL_TEXTURE_3D);

      char buf[256];

      sprintf(buf, " %ld  %d   ra=%.1f  rr=%.1f  rb=%.1f  dt=%.3f   %d %d %d ",
              tima - tim, (int)mode, ra, rr, rb, dt, (int)sigmode, (int)sigtype,
              (int)mixtype);
      drawtext(2, buf);

      sprintf(buf, " b1=%.3f  b2=%.3f  d1=%.3f  d2=%.3f   sn=%.3f  sm=%.3f ",
              b1, b2, d1, d2, sn, sm);
      drawtext(1, buf);

      if (savedispcnt > 0.0)  // if there's a message, display it
      {
        drawtext(0, dispmessage);
        savedispcnt -= (double)(tima - tim) / 1000.0;
        if (savedispcnt < 0.0) savedispcnt = 0.0;
      }
    }

    glBindFramebuffer(GL_FRAMEBUFFER, 0);

    SDL_GL_SwapBuffers();
  }

ende:  // program ending, free all
  TTF_CloseFont(font);
  TTF_Quit();
  if (ttd) {
    delShaders();
    delete_buffers();
  }
  SDL_Quit();
  if (logfile) {
    fprintf(logfile, "ending\n");
    fflush(logfile);
    fclose(logfile);
  }
  return 0;
}
