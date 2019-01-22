from matplotlib import pyplot as plt

def visualize_images(legend, *args):
    if(len(args) == 0):
        return
    
    nrows, ncols = len(args), len(args[0])   
    fig, axs = plt.subplots(nrows=nrows, ncols=ncols)    
    if nrows == 1:
        for row, images in enumerate(args):
            for col in range(ncols):
                axs[col].imshow(images[col], cmap=plt.cm.gray)
            return
        
    for row, images in enumerate(args): # for each bunch of objects
        for col, ax in enumerate(axs[row]): # for each object show it on correct row and col
            if (row == 0):
                ax.set_title(legend[col])
                
            ax.imshow(images[col], cmap=plt.cm.gray)
            ax.set_xticks([])

def visualize_one(title, image):
    plt.title(title)
    plt.imshow(image)

def visualize_pairs(*pairs):
    visualize_images(("lr", "hr"), *pairs)
    
def visualize_triplets(*triplets):
    visualize_images(("lr", "prediction", "hr"), *triplets)
