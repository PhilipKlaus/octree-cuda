
if __name__ == "__main__":

    max_level=3
    path=r".\cmake-build-release\export"

    import os

    _, _, filenames = next(os.walk(path))

    for filename in filenames:
        if filename.endswith(".ply") and len(filename) >= (5 + max_level):
            file_to_delete = os.path.join(path,filename)
            os.remove(file_to_delete)
            print("deleted: {}".format(file_to_delete))