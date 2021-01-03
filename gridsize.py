def calc_grid_index(minimum, maximum, offset, grid, coords):

    
    sizeX = maximum[0] - minimum[0]
    sizeY = maximum[1] - minimum[1]
    sizeZ = maximum[2] - minimum[2]
    
    print(sizeX, sizeY, sizeZ)
    
    stepX = sizeX / grid
    stepY = sizeY / grid
    stepZ = sizeZ / grid
    
    print(stepX, stepY, stepZ)
    
    #ix = int((coords[0] - minimum[0]) / stepX)
    #iy = int((coords[1] - minimum[1]) / stepY)
    #iz = int((coords[2] - minimum[2]) / stepZ)
    
    ux = (coords[0] - minimum[0]) / stepX
    uy = (coords[1] - minimum[1]) / stepY
    uz = (coords[2] - minimum[2]) / stepZ
    
    print(ux, uy, uz)
    
    ix = int(min(ux, grid - 1))
    iy = int(min(uy, grid - 1))
    iz = int(min(uz, grid - 1))
    
    print(ix, iy, iz)

    return ix + iy * grid + iz * grid * grid;
    
if __name__ == "__main__":

    minimum = [-10.0, -10.0, -10.0]
    maximum = [0, 0, 0]
    offset = [1,1,1]

    grid = 128

    result = calc_grid_index(minimum, maximum, offset, grid, [-0, -10, -10])
    print(result)