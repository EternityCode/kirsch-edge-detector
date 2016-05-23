__kernel void kirsch_edges(__global const uchar* conv_table, __global const uchar* colour_map,
    __global uchar3* img_edge)
{
    int x = get_global_id(0);
    int y = get_global_id(1);
    img_edge[y*2560+x].x = 255;
    img_edge[y*2560+x].y = 255;
    img_edge[y*2560+x].z = 255;
}
