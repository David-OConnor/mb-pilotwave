"""
    wave2d.py

    Solve the 2D wave equation on a rectangle and make a movie of the solution

    Copyright (c) 2013 Greg von Winckel

    Permission is hereby granted, free of charge, to any person obtaining
    a copy of this software and associated documentation files (the
    "Software"), to deal in the Software without restriction, including
    without limitation the rights to use, copy, modify, merge, publish,
    distribute, sublicense, and/or sell copies of the Software, and to
    permit persons to whom the Software is furnished to do so, subject to
    the following conditions:

    The above copyright notice and this permission notice shall be
    included in all copies or substantial portions of the Software.

    THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND,
    EXPRESS OR IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF
    MERCHANTABILITY, FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT.
    IN NO EVENT SHALL THE AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY
    CLAIM, DAMAGES OR OTHER LIABILITY, WHETHER IN AN ACTION OF CONTRACT,
    TORT OR OTHERWISE, ARISING FROM, OUT OF OR IN CONNECTION WITH THE
    SOFTWARE OR THE USE OR OTHER DEALINGS IN THE SOFTWARE.

    Created on Sun Nov  3 17:20:21 MST 2013

"""



import matplotlib.pyplot as plt
import numpy as np
import matplotlib.animation as animation


# Final time
T = 2

# Domain dimensions
height = 2
width = 4

# Wave speed
c = 1

# Number of time steps
nt = 400

# Grid points in x direction
nx = 250

# Grid points in y direction
ny = 125

class wave2d(object):
    def __init__(self,height,width,T,nx,ny,nt,c):

        self.x = np.linspace(-0.5*width,0.5*width,nx)
        self.y = np.linspace(-0.5*height,0.5*height,ny)
        self.t = np.linspace(0,T,nt+1)

        self.dx = self.x[1]-self.x[0]
        self.dy = self.y[1]-self.y[0]
        self.dt = self.t[1]-self.t[0]

        self.xx,self.yy = np.meshgrid(self.x,self.y)

        # Gamma_x squared
        self.gx2 = c*self.dt/self.dx

        # Gamma_y squared
        self.gy2 = c*self.dt/self.dy

        # 2*(1-gamma_x^2-gamma_y^2)
        self.gamma = 2*(1 - self.gx2 - self.gy2)


    def solve(self,ffun,gfun):

        f = ffun(self.xx,self.yy)
        g = gfun(self.xx,self.yy)

        u = np.zeros((ny,nx,nt+1))

        # Set initial condition
        u[:,:,0] = f

        """ Compute first time step """
        u[:,:,1] = 0.5*self.gamma*f+self.dt*g
        u[1:-1,1:-1,1] += 0.5*self.gx2*(f[1:-1,2:]+f[1:-1,:-2])
        u[1:-1,1:-1,1] += 0.5*self.gy2*(f[:-2,1:-1]+f[2:,1:-1])

        for k in range(1,nt):
            # Every point contains these terms
            u[:,:,k+1] = self.gamma*u[:,:,k] - u[:,:,k-1]

            # Interior points
            u[1:-1,1:-1,k+1] += self.gx2*(u[1:-1,2:,k]+u[1:-1,:-2,k]) + \
                                self.gy2*(u[2:,1:-1,k]+u[:-2,1:-1,k])

            # Top boundary
            u[0,1:-1,k+1] +=  2*self.gy2*u[1,1:-1,k] + \
                                self.gx2*(u[0,2:,k]+u[0,:-2,k])

            # Right boundary
            u[1:-1,-1,k+1] += 2*self.gx2*u[1:-1,-2,k] + \
                                self.gy2*(u[2:,-1,k]+u[:-2,-1,k])

            # Bottom boundary
            u[-1,1:-1,k+1] +=  2*self.gy2*u[-2,1:-1,k] + \
                                 self.gx2*(u[-1,2:,k]+u[-1,:-2,k])

            # Left boundary
            u[1:-1,0,k+1] += 2*self.gx2*u[1:-1,1,k] + \
                               self.gy2*(u[2:,0,k]+u[:-2,0,k])

            # Top right corner
            u[0,-1,k+1] += 2*self.gx2*u[0,-2,k] + \
                           2*self.gy2*u[1,-1,k]

            # Bottom right corner
            u[-1,-1,k+1] += 2*self.gx2*u[-1,-2,k] + \
                            2*self.gy2*u[-2,-1,k]

            # Bottom left corner
            u[-1,0,k+1] += 2*self.gx2*u[-1,1,k] + \
                           2*self.gy2*u[-2,0,k]

            # Top left corner
            u[0,0,k+1] += 2*self.gx2*u[0,1,k] + \
                          2*self.gy2*u[1,0,k]

        return u

def main():
    wave_eq = wave2d(height, width, T, nx, ny, nt, c)

    # Initial value functions
    f = lambda x, y: np.exp(-10 * (x ** 2 + y ** 2))
    g = lambda x, y: 0

    u = wave_eq.solve(f, g)

    x = wave_eq.x
    y = wave_eq.y

    frames = []
    fig = plt.figure(1,(16,8))

    for k in range(nt+1):
        frame = plt.imshow(u[:,:,k],extent=[x[0],x[-1],y[0],y[-1]])
    frames.append([frame])

    ani = animation.ArtistAnimation(fig,frames,interval=50,
                         blit=True,repeat_delay=1000)
                         #    ani.save('wave2d.mp4',dpi=300)
    plt.show()



if __name__ == '__main__':
    main()
