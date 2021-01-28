
if __name__ == "__main__":

    max_level=3
    path=r".\cmake-build-release\export"

    import os
    import shutil

    _, _, filenames = next(os.walk(path))

    for filename in filenames:
        if filename.endswith(".ply"):
            source_file = os.path.join(path, filename)
            target_folder = f"level_{len(filename)-4}"
            target_folder = os.path.join(path, target_folder)
            if not os.path.exists(target_folder):
                os.makedirs(target_folder)
            shutil.move(source_file, target_folder)
            
            