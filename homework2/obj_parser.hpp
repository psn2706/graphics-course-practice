#pragma once

#include <array>
#include <vector>
#include <map>
#include <filesystem>

#include "stb_image.h"

struct obj_data
{
    std::vector<std::uint32_t> indices;
    std::string tex_name;
    bool texture_added = false;
};

struct tex_data
{
    struct image_data
    {
        stbi_uc * data = nullptr;
        int x = 0, y = 0, channels = 0;
        bool exists = false;
    };

    image_data albedo;
    image_data alpha;
    std::array<float, 3> glossiness = {0, 0, 0};
    float power = 0;
};

struct scene_data
{
    struct vertex
    {
        std::array<float, 3> position;
        std::array<float, 3> normal;
        std::array<float, 2> texcoord;
    };

    std::vector<vertex> vertices;
    std::map<std::string, tex_data*> tex_dict;
    std::vector<obj_data*> objects;
};

scene_data parse_scene(std::filesystem::path const & path, std::string const & project_root);