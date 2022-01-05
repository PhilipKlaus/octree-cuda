import os
import json
import sys
import struct
import argparse
from functools import partial


class HierarchyNode:

    def __init__(self, byte_data):
        self._type = byte_data[0]
        self._bitmask = byte_data[1]
        self._points = byte_data[2]
        self._byte_offset = byte_data[3]
        self._byte_size = byte_data[4]
    
    def __str__(self):
        return f"Bitmask: {self._bitmask}\nPoints: {self._points}\nByteOffset: {self._byte_offset}\nByteSize: {self._byte_size}\nChildren: {self.children()}"
        
    def children(self):
        return bin(self._bitmask).count("1")
        
    def offset(self):
        return self._byte_offset
        
    def size(self):
        return self._byte_size
        

def parse_arguments(argv):
    parser = argparse.ArgumentParser(prog="potree_to_ply", description='Convert Potree data to PLY files.')
    parser.add_argument('-i', type=str, help='potree input data directory')
    parser.add_argument('-o', type=str, help='ply output data directory')
    args = parser.parse_args()
    
    potree_path = args.i
    output_path = args.o if args.o is not None else potree_path
    
    if not potree_path:
        parser.print_help()
        exit()
        
    return potree_path, output_path
    

if __name__ == "__main__":
    
    path, output_path = parse_arguments(sys.argv)
    
    struct_fmt = '=BBIQQ'
    struct_len = struct.calcsize(struct_fmt)
    struct_unpack = struct.Struct(struct_fmt).unpack_from

    hierarchy_file = os.path.join(path, "hierarchy.bin")
    octree_file = os.path.join(path, "octree.bin")
    metadata_file = os.path.join(path, "metadata.json")
    
    with open(octree_file, "rb") as octree:
        with open(metadata_file) as f:
            metadata = json.load(f)
            
            points = metadata["points"]
            hierarchy = metadata["hierarchy"]
            chunk_size = hierarchy["firstChunkSize"]
            step_size = hierarchy["stepSize"]
            depth = hierarchy["depth"]
            node_amount = chunk_size / 22

            with open(hierarchy_file, "rb") as h:
                hierarchy = [HierarchyNode(struct_unpack(chunk)) for chunk in iter(partial(h.read, struct_len), b'')]
               
            index = 0
            visited = [False] * int(node_amount)
               
            # Create a queue for BFS
            queue = []
     
            # Mark the source node as
            # visited and enqueue it
            queue.append((hierarchy[0], 0))
            visited[index] = True
     
            points_to_export = []
            data_to_export = []
            leaf_node_data = []
            leaf_node_points = 0
            
            while queue:
                node, level = queue.pop(0)
                
                if len(points_to_export) == level:
                    points_to_export.append(0)
                    data_to_export.append([])
                    
                # Fetch data from octree 
                octree.seek(node.offset(), 0)
                data = octree.read(node.size())
                
                # Store data and update
                points_to_export[level] += int(node.size() / 18)
                data_to_export[level].append(data)
                
                if node.children() == 0:
                    leaf_node_points += int(node.size() / 18)
                    leaf_node_data.append(data)
                
                for _ in range(node.children()):
                    index += 1
                    if visited[index] == False:
                        queue.append((hierarchy[index], level + 1))
                        visited[index] = True
         
            
            points_to_export.append(leaf_node_points)
            data_to_export.append(leaf_node_data)
            
            for level in range(len(points_to_export)):
            
                header = f'''ply
format binary_little_endian 1.0
comment Created with PotreeToPly Converter
element vertex {points_to_export[level]}
property uint x
property uint y
property uint z
property ushort red
property ushort green
property ushort blue
end_header
'''
                ply_file = os.path.join(output_path, f"converted_level_{level if level < (len(points_to_export) - 1) else 'leaf'}.ply")

                with open(ply_file, "wb") as out:
                    out.write(header.encode("ascii"))
                    for p in data_to_export[level]:
                        out.write(p)
                print(f"Exported points: {points_to_export[level]}")
