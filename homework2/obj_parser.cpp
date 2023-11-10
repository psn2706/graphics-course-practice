#include "obj_parser.hpp"

#include <string>
#include <sstream>
#include <fstream>
#include <stdexcept>
#include <map>

namespace
{

    template <typename ... Args>
    std::string to_string(Args const & ... args)
    {
        std::ostringstream os;
        (os << ... << args);
        return os.str();
    }

}

std::map<std::string, tex_data*> open_mtl(std::filesystem::path const & path, std::string const & project_root)
{
    std::map<std::string, tex_data*> tex_dict;
    tex_data* new_texture;
    std::string name;
    bool compiles_now = false;

    std::ifstream is(path);

    std::string line;
    std::size_t line_count = 0;

    while (std::getline(is >> std::ws, line)) {
        ++line_count;

        if (line.empty()) continue;

        if (line[0] == '#') continue;

        std::istringstream ls(std::move(line));

        std::string tag;
        ls >> tag;

        if (!compiles_now && tag == "newmtl") {
            ls >> name;
            new_texture = new tex_data;
            compiles_now = true;
        }
        else if (!compiles_now) {
            continue;
        }

        else if (tag == "newmtl") {
            tex_dict[name] = new_texture;

            ls >> name;
            new_texture = new tex_data;
        }
        else if (tag == "map_Ka") {
            std::string image_path;
            ls >> image_path;

            image_path = project_root + image_path;
            new_texture->albedo.data = stbi_load(image_path.data(),
                                                 &new_texture->albedo.x, &new_texture->albedo.y,
                                                 &new_texture->albedo.channels, 4);
            new_texture->albedo.exists = true;
        }
        else if (tag == "map_d") {
            std::string image_path;
            ls >> image_path;

            image_path = project_root + image_path;
            new_texture->alpha.data = stbi_load(image_path.data(),
                                                 &new_texture->alpha.x, &new_texture->alpha.y,
                                                 &new_texture->alpha.channels, 4);
            new_texture->alpha.exists = true;
        }
        else if (tag == "Ks") {
            auto & g = new_texture->glossiness;
            ls >> g[0] >> g[1] >> g[2];
        }
        else if (tag == "Ns") {
            ls >> new_texture->power;
        }
    }

    if (compiles_now) {
        tex_dict[name] = new_texture;
    }

    return tex_dict;
}

scene_data parse_scene(std::filesystem::path const & path, std::string const & project_root)
{
    std::ifstream is(path);

    std::vector<std::array<float, 3>> positions;
    positions.push_back({0, 0, 0});
    std::vector<std::array<float, 3>> normals;
    normals.push_back({0, 0, 0});
    std::vector<std::array<float, 2>> texcoords;
    texcoords.push_back({0, 0});

    std::map<std::array<std::uint32_t, 3>, std::uint32_t> index_map;
    std::vector<std::array<std::uint32_t, 3>> index_vec;

    std::map<std::string, tex_data*> tex_dict;

    std::vector<obj_data*> objects;
    obj_data* current = new obj_data;

    std::string line;
    std::size_t line_count = 0;

    auto fail = [&](auto const & ... args){
        throw std::runtime_error(to_string("Error parsing OBJ data, line ", line_count, ": ", args...));
    };

    while (std::getline(is >> std::ws, line))
    {
        ++line_count;

        if (line.empty()) continue;

        if (line[0] == '#') continue;

        std::istringstream ls(std::move(line));

        std::string tag;
        ls >> tag;

        if (tag == "mtllib")
        {
            std::string mtl_path;
            ls >> mtl_path;

            mtl_path = project_root + "/" + mtl_path;
            tex_dict = open_mtl(mtl_path, project_root);
        }
        else if (tag == "v")
        {
            auto & p = positions.emplace_back();
            ls >> p[0] >> p[1] >> p[2];
        }
        else if (tag == "vn")
        {
            auto & n = normals.emplace_back();
            ls >> n[0] >> n[1] >> n[2];
        }
        else if (tag == "vt")
        {
            auto & t = texcoords.emplace_back();
            ls >> t[0] >> t[1];
            t[1] = 1 - t[1];
        }
        else if (tag == "g" && !current->indices.empty())
        {
            objects.push_back(current);
            current = new obj_data;
        }
        else if (tag == "usemtl" && !current->indices.empty())
        {
            objects.push_back(current);
            current = new obj_data;
            ls >> current->tex_name;
            current->texture_added = true;
        }
        else if (tag == "usemtl")
        {
            ls >> current->tex_name;
            current->texture_added = true;
        }
        else if (tag == "f")
        {
            std::vector<std::uint32_t> vertices;

            while (ls)
            {
                std::array<std::int32_t, 3> index{0, 0, 0};
                bool has_texcoord = false;
                bool has_normal = false;

                ls >> index[0];
                if (ls.eof()) break;
                if (!ls)
                    fail("expected position index");

                if (!std::isspace(ls.peek()) && !ls.eof())
                {
                    if (ls.get() != '/')
                        fail("expected '/'");

                    if (ls.peek() != '/')
                    {
                        ls >> index[1];
                        if (!ls)
                            fail("expected texcoord index");
                        has_texcoord = true;

                        if (!std::isspace(ls.peek()) && !ls.eof())
                        {
                            if (ls.get() != '/')
                                fail("expected '/'");

                            ls >> index[2];
                            if (!ls)
                                fail("expected normal index");
                            has_normal = true;
                        }
                    }
                    else
                    {
                        ls.get();

                        ls >> index[2];
                        if (!ls)
                            fail("expected normal index");
                        has_normal = true;
                    }
                }

                if (index[0] < 0)
                    index[0] = positions.size() + index[0];

                if (has_texcoord)
                {
                    if (index[1] < 0)
                        index[1] = texcoords.size() + index[1];
                }
                else
                    index[1] = 0;

                if (has_normal)
                {
                    if (index[2] < 0)
                        index[2] = normals.size() + index[2];
                }
                else
                    index[2] = 0;

                if (index[0] >= positions.size())
                    fail("bad position index (", index[0], ")");

                if (index[1] != -1 && index[1] >= texcoords.size())
                    fail("bad texcoord index (", index[1], ")");

                if (index[2] != -1 && index[2] >= normals.size())
                    fail("bad normal index (", index[2], ")");

                std::array<std::uint32_t, 3> u_index{(unsigned int) index[0],
                                                     (unsigned int) index[1],
                                                     (unsigned int) index[2]};
                auto it = index_map.find(u_index);
                if (it == index_map.end())
                {
                    it = index_map.insert({u_index, index_vec.size()}).first;
                    index_vec.push_back(u_index);

                    /*auto & v = result.vertices.emplace_back();

                    v.position = positions[index[0]];

                    if (index[1] != -1)
                        v.texcoord = texcoords[index[1]];
                    else
                        v.texcoord = {0.f, 0.f};

                    if (index[2] != -1)
                        v.normal = normals[index[2]];
                    else
                        v.normal = {0.f, 0.f, 0.f};*/
                }

                vertices.push_back(it->second);
            }

            for (std::size_t i = 1; i + 1 < vertices.size(); ++i)
            {
                current->indices.push_back(vertices[0]);
                current->indices.push_back(vertices[i]);
                current->indices.push_back(vertices[i + 1]);
            }
        }
    }

    if (!current->indices.empty())
        objects.push_back(current);

    std::vector<scene_data::vertex> ret_vertices;
    for (auto it : index_vec)
        ret_vertices.push_back({positions[it[0]], normals[it[2]], texcoords[it[1]]});

    return {ret_vertices, tex_dict, objects};
}
