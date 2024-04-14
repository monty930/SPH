// ----------------------------------------------------------------------------
// main.cpp
//
//  Created on: Fri Jan 22 20:45:07 2021
//      Author: Kiwon Um
//        Mail: kiwon.um@telecom-paris.fr
//
// Description: SPH simulator (DO NOT DISTRIBUTE!)
//
// Copyright 2021-2024 Kiwon Um
//
// The copyright to the computer program(s) herein is the property of Kiwon Um,
// Telecom Paris, France. The program(s) may be used and/or copied only with
// the written permission of Kiwon Um or in accordance with the terms and
// conditions stipulated in the agreement/contract under which the program(s)
// have been supplied.
// ----------------------------------------------------------------------------

#define _USE_MATH_DEFINES

#include <glad/glad.h>
#include <GLFW/glfw3.h>

#include <iostream>
#include <sstream>
#include <iomanip>
#include <vector>
#include <cmath>

#ifndef M_PI
#define M_PI 3.141592
#endif

#include "Vector.hpp"

// window parameters
GLFWwindow *gWindow = nullptr;
int gWindowWidth = 1024;
int gWindowHeight = 768;

// timer
float gAppTimer = 0.0;
float gAppTimerLastClockTime;
bool gAppTimerStoppedP = true;

// global options
bool gPause = true;
bool gSaveFile = false;
bool gShowGrid = true;
bool gShowVel = false;
int gSavedCnt = 0;

const int kViewScale = 15;

// SPH Kernel function: cubic spline
class CubicSpline
{
public:
  explicit CubicSpline(const Real h = 1)
  {
    setSmoothingLen(h);
  }
  void setSmoothingLen(const Real h)
  {
    const Real h2 = square(h), h3 = h2 * h;
    _h = h;
    _sr = 2e0 * h; // was: 2
  }
  Real supportRadius() const { return _sr; }

private:
  unsigned int _dim;
  Real _h, _sr;
};

class SphSolver
{
public:
  explicit SphSolver(
      const Real nu = 0.08, const Real h = 0.5, const Real density = 1e3,
      const Vec2f g = Vec2f(0, -9.8), const Real eta = 0.01, const Real gamma = 7.0) : _kernel(h), _nu(nu), _h(h), _d0(density),
                                                                                       _g(g), _eta(eta), _gamma(gamma)
  {
    _dt = 0.0005;
    _c = std::fabs(_g.y) / _eta;
    _k = _d0 * _c * _c / _gamma;
  }

  // assume an arbitrary grid with the size of res_x*res_y; a fluid mass fill up
  // the size of f_width, f_height; each cell is sampled with 2x2 particles.
  void initScene(
      const int res_x, const int res_y, const int f_width, const int f_height)
  {
    _pos.clear();

    _resX = res_x;
    _resY = res_y;

    // set wall for boundary
    _l = 0.5 * _h;
    _r = static_cast<Real>(res_x) - 0.5 * _h;
    _b = 0.5 * _h;
    _t = static_cast<Real>(res_y) - 0.5 * _h;

    // sample a fluid mass
    for (int j = 0; j < f_height; ++j)
    {
      for (int i = 0; i < f_width; ++i)
      {
        _pos.push_back(Vec2f(i + 0.25, j + 0.25));
        _pos.push_back(Vec2f(i + 0.75, j + 0.25));
        _pos.push_back(Vec2f(i + 0.25, j + 0.75));
        _pos.push_back(Vec2f(i + 0.75, j + 0.75));
      }
    }

    // copy _pos into _prev_pos
    _prev_pos = _pos;

    // make sure for the other particle quantities
    _vel = std::vector<Vec2f>(_pos.size(), Vec2f(0, 0));
    _p = std::vector<Real>(_pos.size(), 0);
    _d = std::vector<Real>(_pos.size(), 0);

    _col = std::vector<float>(_pos.size() * 4, 1.0); // RGBA
    _vln = std::vector<float>(_pos.size() * 4, 0.0); // GL_LINES
  }

  void update()
  {
    std::cout << '.' << std::flush;

    buildNeighbor();
  
    // simulation step based on related paper (algorithm 1)
    for (tIndex i = 0; i < particleCount(); ++i)
    {
      // apply gravity
      _vel[i] += _dt * _g;
    }
    // modify velocities with pairwise viscosity impulses
    applyViscosity();
    for (tIndex i = 0; i < particleCount(); ++i)
    {
      // save previous position
      _prev_pos[i] = _pos[i];
      // advance to predicted position
      _pos[i] += _dt * _vel[i];
    }
    // add and remove springs, change rest lengths
    adjustSprings();
    // modify positions according to springs,
    // double density relaxation, and collisions
    applySpringDisplacements();
    doubleDensityRelaxation();
    for (tIndex i = 0; i < particleCount(); ++i)
    {
      // use previous position to compute next velocity
      _vel[i] = (_pos[i] - _prev_pos[i]) / _dt;
    }
    resolveCollisions();

    if (gShowVel)
      updateVelLine();
  }

  tIndex particleCount() const { return _pos.size(); }
  const Vec2f &position(const tIndex i) const { return _pos[i]; }
  const float &color(const tIndex i) const { return _col[i]; }
  const float &vline(const tIndex i) const { return _vln[i]; }

  int resX() const { return _resX; }
  int resY() const { return _resY; }

private:
  void buildNeighbor()
  {
    _pidxInGrid.clear();
    _pidxInGrid.resize(_resX * _resY);
    int count = 0;

    for (tIndex i = 0; i < particleCount(); ++i)
    {
      const int gx = static_cast<int>(_pos[i].x);
      const int gy = static_cast<int>(_pos[i].y);

      if (gx >= 0 && gx < _resX && gy >= 0 && gy < _resY)
      {
        _pidxInGrid[idx1d(gx, gy)].push_back(i);
        count++;
      }
    }
    // std::cout << "count: " << count << std::endl;
  }

  std::tuple<int, int, int, int> getNeighbourBounds(const tIndex i) {
    const int supportRadius = static_cast<int>(_kernel.supportRadius());
    const int gx = static_cast<int>(_pos[i].x);
    const int gy = static_cast<int>(_pos[i].y);

    const int sx1 = std::max(0, gx - supportRadius);
    const int sx2 = std::min(_resX - 1, gx + supportRadius + 1);
    const int sy1 = std::max(0, gy - supportRadius);
    const int sy2 = std::min(_resY - 1, gy + supportRadius + 1);
    return std::make_tuple(sx1, sx2, sy1, sy2);
  }

  void applyViscosity() {
    //TODO
  }

  void adjustSprings() {
    //TODO
  }

  void applySpringDisplacements() {
    //TODO
  }

  void doubleDensityRelaxation() {
    const int rad = static_cast<int>(_kernel.supportRadius()); // support radius (in paper: h)
    for (tIndex i = 0; i < particleCount(); ++i)
    {
      Real rho = 0;
      Real rho_near = 0;
      Vec2f dx = Vec2f(0, 0);
      
      std::tuple<int, int, int, int> bounds = getNeighbourBounds(i);
      const int sx1 = std::get<0>(bounds), sx2 = std::get<1>(bounds);
      const int sy1 = std::get<2>(bounds), sy2 = std::get<3>(bounds);
      for (int gx = sx1; gx <= sx2; ++gx)
      {
        for (int gy = sy1; gy <= sy2; ++gy)
        {
          const std::vector<tIndex> &neighbors = _pidxInGrid[idx1d(gx, gy)];
          for (tIndex j : neighbors) // for each particle j in neighbours(i)
          {
            if (i == j)
              continue;
            const Vec2f rij = _pos[j] - _pos[i];
            const Real r = rij.length();
            // compute density and near-density
            if (r < rad)
            {
              const Real q = 1. - r / rad;
              rho += q * q;
              rho_near += q * q * q;
            }
          }
          // std::cout << "rho: " << rho << std::endl;
          // std::cout << "rho_near: " << rho_near << std::endl;

          const Real k = 1.0;
          const Real k_near = 1.0;
          const Real rho_0 = 2.0;

          // std::cout << "rad, k, k_near, rho_0: " << rad << " " << k << " " << k_near << " " << rho_0 << std::endl;
          
          // compute pressure and near-pressure
          Real press = k * (rho - rho_0); // TODO CHECK _k an _d0
          if (press < 0.0)
            press = 0.0;
          // const Real press_near = k_near * rho_near;
          const Real press_near = 0.0;
          // std::cout << "press: " << press << std::endl;
          // std::cout << "press_near: " << press_near << std::endl;
          dx = Vec2f(0, 0);
          for (tIndex j : neighbors) // for each particle j in neighbours(i)
          {
            if (i == j)
              continue;
            const Vec2f rij = _pos[j] - _pos[i];
            const Real r = rij.length();
            // std::cout << "r: " << r << std::endl;
            // exit(1);
            if (r < rad)
            {
              // apply displacements
              const Real q = 1 - r / rad;
              const Vec2f rij1 = rij / r; // unit vector from particle i to j
              const Vec2f D = _dt * _dt * (press * q + press_near * q * q) * rij1;
              const Vec2f D2 = D / 2;
              _pos[j] += D2;
              dx -= D2;
            }
          }
          _pos[i] += dx;
          // if (dx.length() > 0.0)
            // std::cout << "dx: " << dx << std::endl;
        }
      }
    }
  }

  // simple collision detection/resolution for each particle
  void resolveCollisions()
  {
    std::vector<tIndex> need_res;
    for (tIndex i = 0; i < particleCount(); ++i)
    {
      if (_pos[i].x < _l || _pos[i].y < _b || _pos[i].x > _r || _pos[i].y > _t)
        need_res.push_back(i);
    }

    for (
        std::vector<tIndex>::const_iterator it = need_res.begin();
        it < need_res.end();
        ++it)
    {
      const Vec2f p0 = _pos[*it];
      _pos[*it].x = clamp(_pos[*it].x, _l, _r);
      _pos[*it].y = clamp(_pos[*it].y, _b, _t);
      _vel[*it] = (_pos[*it] - p0) / _dt;
    }
  }

  void updateVelLine()
  {
    for (tIndex i = 0; i < particleCount(); ++i)
    {
      _vln[i * 4 + 0] = _pos[i].x;
      _vln[i * 4 + 1] = _pos[i].y;
      _vln[i * 4 + 2] = _pos[i].x + _vel[i].x;
      _vln[i * 4 + 3] = _pos[i].y + _vel[i].y;
    }
  }

  inline tIndex idx1d(const int i, const int j) { return i + j * resX(); }

  const CubicSpline _kernel;

  // particle data
  std::vector<Vec2f> _pos; // position
  std::vector<Vec2f> _prev_pos; // previous position
  std::vector<Vec2f> _vel; // velocity
  std::vector<Real> _p;    // pressure
  std::vector<Real> _d;    // density

  std::vector<std::vector<tIndex>> _pidxInGrid; // will help you find neighbor particles

  std::vector<float> _col; // particle color; just for visualization
  std::vector<float> _vln; // particle velocity lines; just for visualization

  // simulation
  Real _dt; // time step

  int _resX, _resY; // background grid resolution

  // wall
  Real _l, _r, _b, _t; // wall (boundary)

  // SPH coefficients
  Real _nu; // viscosity coefficient
  Real _d0; // rest density
  Real _h;  // particle spacing (i.e., diameter)
  Vec2f _g; // gravity

  Real _k;  // EOS coefficient

  Real _eta;
  Real _c;     // speed of sound
  Real _gamma; // EOS power factor
};

SphSolver gSolver(0.08, 0.5, 1e3, Vec2f(0, -9.8), 0.01, 7.0);

void printHelp()
{
  std::cout << "> Help:" << std::endl
            << "    Keyboard commands:" << std::endl
            << "    * H: print this help" << std::endl
            << "    * P: toggle simulation" << std::endl
            << "    * G: toggle grid rendering" << std::endl
            << "    * V: toggle velocity rendering" << std::endl
            << "    * S: save current frame into a file" << std::endl
            << "    * Q: quit the program" << std::endl;
}

// Executed each time the window is resized. Adjust the aspect ratio and the rendering viewport to the current window.
void windowSizeCallback(GLFWwindow *window, int width, int height)
{
  gWindowWidth = width;
  gWindowHeight = height;
  glViewport(0, 0, static_cast<GLint>(gWindowWidth), static_cast<GLint>(gWindowHeight));
  glMatrixMode(GL_PROJECTION);
  glLoadIdentity();
  glOrtho(0, gSolver.resX(), 0, gSolver.resY(), 0, 1);
}

// Executed each time a key is entered.
void keyCallback(GLFWwindow *window, int key, int scancode, int action, int mods)
{
  if (action == GLFW_PRESS && key == GLFW_KEY_H)
  {
    printHelp();
  }
  else if (action == GLFW_PRESS && key == GLFW_KEY_S)
  {
    gSaveFile = !gSaveFile;
  }
  else if (action == GLFW_PRESS && key == GLFW_KEY_G)
  {
    gShowGrid = !gShowGrid;
  }
  else if (action == GLFW_PRESS && key == GLFW_KEY_V)
  {
    gShowVel = !gShowVel;
  }
  else if (action == GLFW_PRESS && key == GLFW_KEY_P)
  {
    gAppTimerStoppedP = !gAppTimerStoppedP;
    if (!gAppTimerStoppedP)
      gAppTimerLastClockTime = static_cast<float>(glfwGetTime());
  }
  else if (action == GLFW_PRESS && key == GLFW_KEY_Q)
  {
    glfwSetWindowShouldClose(window, true);
  }
}

void initGLFW()
{
  // Initialize GLFW, the library responsible for window management
  if (!glfwInit())
  {
    std::cerr << "ERROR: Failed to init GLFW" << std::endl;
    std::exit(EXIT_FAILURE);
  }

  // Before creating the window, set some option flags
  glfwWindowHint(GLFW_CONTEXT_VERSION_MAJOR, 2);
  glfwWindowHint(GLFW_CONTEXT_VERSION_MINOR, 0);
  // glfwWindowHint(GLFW_OPENGL_FORWARD_COMPAT, GL_TRUE); // only if requesting 3.0 or above
  // glfwWindowHint(GLFW_OPENGL_PROFILE, GLFW_OPENGL_CORE_PROFILE);
  glfwWindowHint(GLFW_OPENGL_PROFILE, GLFW_OPENGL_ANY_PROFILE); // for OpenGL below 3.2
  glfwWindowHint(GLFW_RESIZABLE, GL_TRUE);

  // Create the window
  gWindowWidth = gSolver.resX() * kViewScale;
  gWindowHeight = gSolver.resY() * kViewScale;
  gWindow = glfwCreateWindow(
      gSolver.resX() * kViewScale, gSolver.resY() * kViewScale,
      "Basic SPH Simulator", nullptr, nullptr);
  if (!gWindow)
  {
    std::cerr << "ERROR: Failed to open window" << std::endl;
    glfwTerminate();
    std::exit(EXIT_FAILURE);
  }

  // Load the OpenGL context in the GLFW window
  glfwMakeContextCurrent(gWindow);

  // not mandatory for all, but MacOS X
  glfwGetFramebufferSize(gWindow, &gWindowWidth, &gWindowHeight);

  // Connect the callbacks for interactive control
  glfwSetWindowSizeCallback(gWindow, windowSizeCallback);
  glfwSetKeyCallback(gWindow, keyCallback);

  std::cout << "Window created: " << gWindowWidth << ", " << gWindowHeight << std::endl;
}

void clear();

void exitOnCriticalError(const std::string &message)
{
  std::cerr << "> [Critical error]" << message << std::endl;
  std::cerr << "> [Clearing resources]" << std::endl;
  clear();
  std::cerr << "> [Exit]" << std::endl;
  std::exit(EXIT_FAILURE);
}

void initOpenGL()
{
  // Load extensions for modern OpenGL
  if (!gladLoadGLLoader((GLADloadproc)glfwGetProcAddress))
    exitOnCriticalError("[Failed to initialize OpenGL context]");

  glDisable(GL_CULL_FACE);
  glDisable(GL_LIGHTING);
  glDisable(GL_DEPTH_TEST);

  glViewport(0, 0, static_cast<GLint>(gWindowWidth), static_cast<GLint>(gWindowHeight));
  glMatrixMode(GL_PROJECTION);
  glLoadIdentity();
  glOrtho(0, gSolver.resX(), 0, gSolver.resY(), 0, 1);
}

void init()
{
  // gSolver.initScene(48, 32, 16, 16);
  gSolver.initScene(20, 20, 10, 10);

  initGLFW(); // Windowing system
  initOpenGL();
}

void clear()
{
  glfwDestroyWindow(gWindow);
  glfwTerminate();
}

// The main rendering call
void render()
{
  glClearColor(.4f, .4f, .4f, 1.0f);
  glClear(GL_COLOR_BUFFER_BIT | GL_DEPTH_BUFFER_BIT);

  // grid guides
  if (gShowGrid)
  {
    glBegin(GL_LINES);
    for (int i = 1; i < gSolver.resX(); ++i)
    {
      glColor3f(0.3, 0.3, 0.3);
      glVertex2f(static_cast<Real>(i), 0.0);
      glColor3f(0.3, 0.3, 0.3);
      glVertex2f(static_cast<Real>(i), static_cast<Real>(gSolver.resY()));
    }
    for (int j = 1; j < gSolver.resY(); ++j)
    {
      glColor3f(0.3, 0.3, 0.3);
      glVertex2f(0.0, static_cast<Real>(j));
      glColor3f(0.3, 0.3, 0.3);
      glVertex2f(static_cast<Real>(gSolver.resX()), static_cast<Real>(j));
    }
    glEnd();
  }

  // render particles
  glEnable(GL_POINT_SMOOTH);
  glEnable(GL_BLEND);
  glBlendFunc(GL_SRC_ALPHA, GL_ONE_MINUS_SRC_ALPHA);

  glEnableClientState(GL_VERTEX_ARRAY);
  glEnableClientState(GL_COLOR_ARRAY);

  glPointSize(0.25f * kViewScale);

  glColorPointer(4, GL_FLOAT, 0, &gSolver.color(0));
  glVertexPointer(2, GL_FLOAT, 0, &gSolver.position(0));
  glDrawArrays(GL_POINTS, 0, gSolver.particleCount());

  glDisableClientState(GL_COLOR_ARRAY);
  glDisableClientState(GL_VERTEX_ARRAY);

  // velocity
  if (gShowVel)
  {
    glColor4f(0.0f, 0.0f, 0.5f, 0.2f);

    glEnableClientState(GL_VERTEX_ARRAY);

    glVertexPointer(2, GL_FLOAT, 0, &gSolver.vline(0));
    glDrawArrays(GL_LINES, 0, gSolver.particleCount() * 2);

    glDisableClientState(GL_VERTEX_ARRAY);
  }

  if (gSaveFile)
  {
    std::stringstream fpath;
    fpath << "s" << std::setw(4) << std::setfill('0') << gSavedCnt++ << ".tga";

    std::cout << "Saving file " << fpath.str() << " ... " << std::flush;
    const short int w = gWindowWidth;
    const short int h = gWindowHeight;
    std::vector<int> buf(w * h * 3, 0);
    glReadPixels(0, 0, w, h, GL_BGR, GL_UNSIGNED_BYTE, &(buf[0]));

    FILE *out = fopen(fpath.str().c_str(), "wb");
    short TGAhead[] = {0, 2, 0, 0, 0, 0, w, h, 24};
    fwrite(&TGAhead, sizeof(TGAhead), 1, out);
    fwrite(&(buf[0]), 3 * w * h, 1, out);
    fclose(out);
    gSaveFile = false;

    std::cout << "Done" << std::endl;
  }
}

// Update any accessible variable based on the current time
void update(const float currentTime)
{
  if (!gAppTimerStoppedP)
  {
    // NOTE: When you want to use application's dt ...
    // const float dt = currentTime - gAppTimerLastClockTime;
    // gAppTimerLastClockTime = currentTime;
    // gAppTimer += dt;

    // solve 30 steps
    for (int i = 0; i < 30; ++i)
      gSolver.update();
  }
}

int main(int argc, char **argv)
{
  init();
  while (!glfwWindowShouldClose(gWindow))
  {
    update(static_cast<float>(glfwGetTime()));
    render();
    glfwSwapBuffers(gWindow);
    glfwPollEvents();
  }
  clear();
  std::cout << " > Quit" << std::endl;
  return EXIT_SUCCESS;
}
