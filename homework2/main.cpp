#ifdef WIN32
#include <SDL.h>
#undef main
#else
#include <SDL2/SDL.h>
#endif

#include <GL/glew.h>

#include <string_view>
#include <stdexcept>
#include <iostream>
#include <chrono>
#include <vector>
#include <map>
#include <cmath>
#include <fstream>
#include <sstream>

#define GLM_FORCE_SWIZZLE
#define GLM_ENABLE_EXPERIMENTAL
#include <glm/vec3.hpp>
#include <glm/mat4x4.hpp>
#include <glm/ext/matrix_transform.hpp>
#include <glm/ext/matrix_clip_space.hpp>
#include <glm/ext/scalar_constants.hpp>
#include <glm/gtx/string_cast.hpp>

#include "obj_parser.hpp"
#include "stb_image.h"
#include "stb_image.c"
//#include "obj_parser.ñpp"
//#include "sponza/sponza.mtl"

std::string to_string(std::string_view str)
{
    return std::string(str.begin(), str.end());
}

void sdl2_fail(std::string_view message)
{
    throw std::runtime_error(to_string(message) + SDL_GetError());
}

void glew_fail(std::string_view message, GLenum error)
{
    throw std::runtime_error(to_string(message) + reinterpret_cast<const char *>(glewGetErrorString(error)));
}

const char vertex_shader_source[] =
R"(#version 330 core

uniform mat4 model;
uniform mat4 view;
uniform mat4 projection;
uniform mat4 lightSpaceMatrix;

layout (location = 0) in vec3 in_position;
layout (location = 1) in vec3 in_normal;
layout (location = 2) in vec2 in_texcoord;

out vec3 position;
out vec3 normal;
out vec2 texcoord;
out vec4 FragPosLightSpace;

void main()
{
    gl_Position = projection * view * model * vec4(in_position, 1.0);
    position = (model * vec4(in_position, 1.0)).xyz;
    normal = normalize((model * vec4(in_normal, 0.0)).xyz);
    texcoord = (model * vec4(in_texcoord, 1.0, 1.0)).xy;
    FragPosLightSpace = lightSpaceMatrix * vec4(position, 1.0);
}
)";

const char fragment_shader_source[] =
R"(#version 330 core

uniform mat4 transform;
uniform sampler2D shadow_map;

uniform vec3 ambient;
uniform vec3 glossiness;
uniform float roughness;

uniform vec3 sun_direction;
uniform vec3 sun_color;
uniform vec3 pl_position;
uniform vec3 pl_color;
uniform vec3 pl_attenuation;

uniform sampler2D albedo_samp;
uniform sampler2D alpha_samp;
uniform int is_albedo;
uniform int is_alpha;

uniform vec3 camera_position;


in vec3 position;
in vec3 normal;
in vec2 texcoord;
in vec4 FragPosLightSpace;

layout (location = 0) out vec4 out_color;

float shadow_bias = 0.0001;

float ShadowCalculation(vec4 fragPosLightSpace)
{
    // perform perspective divide
    vec3 projCoords = fragPosLightSpace.xyz / fragPosLightSpace.w;
    // transform to [0,1] range
    projCoords = projCoords * 0.5 + 0.5;
    // get closest depth value from light's perspective (using [0,1] range fragPosLight as coords)
    float closestDepth = texture(shadow_map, projCoords.xy).r; 
    // get depth of current fragment from light's perspective
    float currentDepth = projCoords.z;
    // check whether current frag pos is in shadow
    float bias = 0.5;
    float shadow = currentDepth - bias > closestDepth  ? 1.0 : 0.0;

    return shadow;
}

vec3 specular(vec3 dir)
{
    vec3 reflected = 2 * normal * dot(normal, dir) - dir;
    vec3 view_dir = normalize(camera_position - position);
    return glossiness * pow(max(0.0, dot(reflected, view_dir)), roughness);
}

float att(float r) {
    return 1 / (pl_attenuation.x + pl_attenuation.y * r + pl_attenuation.z * r * r);
}

vec3 diffuse(vec3 dir)
{
    return max(0.0, dot(normal, dir)) + specular(dir);
}

float calc_shadow() {
    vec4 shadow_pos = transform * vec4(position, 1.0);
    shadow_pos /= shadow_pos.w;
    shadow_pos = shadow_pos * 0.5 + vec4(0.5);

    bool in_shadow_texture = (shadow_pos.x > 0.0) && (shadow_pos.x < 1.0) && (shadow_pos.y > 0.0) && (shadow_pos.y < 1.0) && (shadow_pos.z > 0.0) && (shadow_pos.z < 1.0);
    float shadow_factor = 1.0;
    if (in_shadow_texture) {
        float sum_n = 0.0;
        float sum_n2 = 0.0;
        float sum_w = 0.0;

        const int N = 5;
        float radius = 3.0;
        for (int x = -N; x <= N; ++x) {
            for (int y = -N; y <= N; ++y) {
                float c = exp(-float(x*x + y*y) / (radius*radius));
                vec2 data = texture(shadow_map, shadow_pos.xy + vec2(x / 2048.0, y / 2048.0)).rg;
                sum_n += c * data.r;
                sum_n2 += c * data.g;
                sum_w += c;
            }
        }

        if (sum_w == 0.0) {
            sum_n = 1;
            sum_n2 = 1;
            sum_w = 1;
        }

        float mu = sum_n / sum_w;
        float sigma = sum_n2 / sum_w - mu * mu;

        float z = shadow_pos.z - shadow_bias;
        shadow_factor = (z < mu) ? 1.0 : sigma / (sigma + (z - mu) * (z - mu));

        float delta = 0.75;
        shadow_factor = (shadow_factor < delta) ? 0.0 : (shadow_factor - delta) / (1 - delta);
    }
    return shadow_factor;
}

void main()
{
    if (is_alpha == 1 && texture(alpha_samp, texcoord).r < 0.5)
        discard;

    float shadow_factor = calc_shadow();    

    vec3 albedo = vec3(1.0, 1.0, 1.0);
    if (is_albedo == 1)
        albedo = texture(albedo_samp, texcoord).rgb;

    vec3 to_point_light = pl_position - position;
    float R = length(to_point_light);

    

    vec3 light = ambient;
    light += sun_color * diffuse(sun_direction) * shadow_factor;
    light += diffuse(to_point_light / R) * pl_color * att(R);
    vec3 color = albedo * light;

    //vec3 __lightColor = vec3(1.0);
    //vec3 __lightDir = normalize(pl_position - position);
    //float __diff = max(dot(lightDir, normal), 0.0);
    //vec3 __light_diffuse = __diff * __lightColor;
    //vec3 __viewDir = normalize(camera_position-position);
    //float __spec = 0.0;
    //vec3 __halfwayDir = normalize(__lightDir + __viewDir);
    //__spec = pow(max(dot(normal, __halfwayDir), 0.0), 64.0);
    //vec3 __specular = __spec * __lightColor;
    //float shadow = ShadowCalculation(FragPosLightSpace);
    //vec3 __lighting = (ambient + (1.0 - shadow) * (diffuse + specular)) * color;
    
    out_color = vec4(color, 1.0);
}
)";

const char shadow_vertex_shader_source[] =
R"(#version 330 core

uniform mat4 model;
uniform mat4 transform;

layout (location = 0) in vec3 in_position;
layout (location = 2) in vec2 in_texcoord;

out vec2 texcoord;

void main()
{
    gl_Position = transform * model * vec4(in_position, 1.0);
    texcoord = (model * vec4(in_texcoord, 1.0, 1.0)).xy;
}
)";

const char shadow_fragment_shader_source[] =
R"(#version 330 core

layout (location = 0) out vec4 out_color;

uniform sampler2D alpha_samp;
uniform int is_alpha;
in vec2 texcoord;

void main()
{
    if (is_alpha == 1 && texture(alpha_samp, texcoord).r < 0.5)
        discard;

    float z = gl_FragCoord.z;
    out_color = vec4(z, z * z + 0.25 * ((dFdx(z) * dFdx(z)) + (dFdy(z) * dFdy(z))), 0.0, 0.0);
}
)";

GLuint create_shader(GLenum type, const char *source)
{
    GLuint result = glCreateShader(type);
    glShaderSource(result, 1, &source, nullptr);
    glCompileShader(result);
    GLint status;
    glGetShaderiv(result, GL_COMPILE_STATUS, &status);
    if (status != GL_TRUE)
    {
        GLint info_log_length;
        glGetShaderiv(result, GL_INFO_LOG_LENGTH, &info_log_length);
        std::string info_log(info_log_length, '\0');
        glGetShaderInfoLog(result, info_log.size(), nullptr, info_log.data());
        throw std::runtime_error("Shader compilation failed: " + info_log);
    }
    return result;
}

GLuint create_program(GLuint vertex_shader, GLuint fragment_shader)
{
    GLuint result = glCreateProgram();
    glAttachShader(result, vertex_shader);
    glAttachShader(result, fragment_shader);
    glLinkProgram(result);

    GLint status;
    glGetProgramiv(result, GL_LINK_STATUS, &status);
    if (status != GL_TRUE)
    {
        GLint info_log_length;
        glGetProgramiv(result, GL_INFO_LOG_LENGTH, &info_log_length);
        std::string info_log(info_log_length, '\0');
        glGetProgramInfoLog(result, info_log.size(), nullptr, info_log.data());
        throw std::runtime_error("Program linkage failed: " + info_log);
    }

    return result;
}

int main(int argc, char **argv) try
{
    if (SDL_Init(SDL_INIT_VIDEO) != 0)
        sdl2_fail("SDL_Init: ");

    SDL_GL_SetAttribute(SDL_GL_CONTEXT_MAJOR_VERSION, 3);
    SDL_GL_SetAttribute(SDL_GL_CONTEXT_MINOR_VERSION, 3);
    SDL_GL_SetAttribute(SDL_GL_CONTEXT_PROFILE_MASK, SDL_GL_CONTEXT_PROFILE_CORE);
    SDL_GL_SetAttribute(SDL_GL_DOUBLEBUFFER, 1);
    SDL_GL_SetAttribute(SDL_GL_RED_SIZE, 8);
    SDL_GL_SetAttribute(SDL_GL_GREEN_SIZE, 8);
    SDL_GL_SetAttribute(SDL_GL_BLUE_SIZE, 8);
    SDL_GL_SetAttribute(SDL_GL_DEPTH_SIZE, 24);

    SDL_Window *window = SDL_CreateWindow("Graphics course homework 2",
        SDL_WINDOWPOS_CENTERED,
        SDL_WINDOWPOS_CENTERED,
        800, 600,
        SDL_WINDOW_OPENGL | SDL_WINDOW_RESIZABLE | SDL_WINDOW_MAXIMIZED);

    if (!window)
        sdl2_fail("SDL_CreateWindow: ");

    SDL_SetWindowFullscreen(window, SDL_WINDOW_FULLSCREEN);
    SDL_ShowCursor(SDL_DISABLE);

    int width, height;
    SDL_GetWindowSize(window, &width, &height);

    int center_x = width / 2, center_y = height / 2;
    SDL_WarpMouseInWindow(window, center_x, center_y);

    SDL_GLContext gl_context = SDL_GL_CreateContext(window);
    if (!gl_context)
        sdl2_fail("SDL_GL_CreateContext: ");

    if (auto result = glewInit(); result != GLEW_NO_ERROR)
        glew_fail("glewInit: ", result);

    if (!GLEW_VERSION_3_3)
        throw std::runtime_error("OpenGL 3.3 is not supported");

    auto vertex_shader = create_shader(GL_VERTEX_SHADER, vertex_shader_source);
    auto fragment_shader = create_shader(GL_FRAGMENT_SHADER, fragment_shader_source);
    auto program = create_program(vertex_shader, fragment_shader);

    // matrix
    GLuint model_location = glGetUniformLocation(program, "model");
    GLuint view_location = glGetUniformLocation(program, "view");
    GLuint projection_location = glGetUniformLocation(program, "projection");
    GLuint transform_location = glGetUniformLocation(program, "transform");
    GLuint lightSpaceMatrix_location = glGetUniformLocation(program, "lightSpaceMatrix");

    // phong params
    GLuint ambient_location = glGetUniformLocation(program, "ambient");
    GLuint albedo_texture_location = glGetUniformLocation(program, "albedo_samp");
    GLuint is_albedo_location = glGetUniformLocation(program, "is_albedo");
    GLuint glossiness_location = glGetUniformLocation(program, "glossiness");
    GLuint roughness_location = glGetUniformLocation(program, "roughness");

    // light sources
    GLuint sun_direction_location = glGetUniformLocation(program, "sun_direction");
    GLuint sun_color_location = glGetUniformLocation(program, "sun_color");
    GLuint pl_position_location = glGetUniformLocation(program, "pl_position");
    GLuint pl_color_location = glGetUniformLocation(program, "pl_color");
    GLuint pl_attenuation_location = glGetUniformLocation(program, "pl_attenuation");

    // another params
    GLuint shadow_map_location = glGetUniformLocation(program, "shadow_map");
    GLuint alpha_texture_location = glGetUniformLocation(program, "alpha_samp");
    GLuint is_alpha_location = glGetUniformLocation(program, "is_alpha");
    GLuint camera_location = glGetUniformLocation(program, "camera_position");

    auto shadow_vertex_shader = create_shader(GL_VERTEX_SHADER, shadow_vertex_shader_source);
    auto shadow_fragment_shader = create_shader(GL_FRAGMENT_SHADER, shadow_fragment_shader_source);
    auto shadow_program = create_program(shadow_vertex_shader, shadow_fragment_shader);

    GLuint shadow_model_location = glGetUniformLocation(shadow_program, "model");
    GLuint shadow_transform_location = glGetUniformLocation(shadow_program, "transform");

    GLuint shadow_alpha_texture_location = glGetUniformLocation(shadow_program, "alpha_samp");
    GLuint shadow_is_alpha_location = glGetUniformLocation(shadow_program, "is_alpha");

    auto scene = parse_scene("sponza/sponza.obj", "sponza/");

    std::vector<std::uint32_t> all_indices;
    std::size_t object_num = scene.objects.size(), pos = 0;
    std::vector<std::pair<std::size_t, std::size_t>> draw_idx(object_num);

    std::size_t i = 0;
    for (auto elem : scene.objects) {
        draw_idx[i].first = pos;
        draw_idx[i].second = elem->indices.size();
        pos += elem->indices.size();
        all_indices.insert(all_indices.end(), elem->indices.begin(), elem->indices.end());
        ++i;
    }

    std::vector<float> xs, ys, zs;
    for (auto v : scene.vertices) {
        xs.push_back(v.position[0]);
        ys.push_back(v.position[1]);
        zs.push_back(v.position[2]);
    }

    float
        x_min = *min_element(xs.begin(), xs.end()),
        x_max = *max_element(xs.begin(), xs.end()),
        x_mid = (x_max + x_min) / 2,
        x_w = x_max - x_min,

        y_min = *min_element(ys.begin(), ys.end()),
        y_max = *max_element(ys.begin(), ys.end()),
        y_mid = (y_max + y_min) / 2,
        y_w = y_max - y_min,

        z_min = *min_element(zs.begin(), zs.end()),
        z_max = *max_element(zs.begin(), zs.end()),
        z_mid = (z_max + z_min) / 2,
        z_w = z_max - z_min;

    GLuint vao, vbo, ebo;
    glGenVertexArrays(1, &vao);
    glBindVertexArray(vao);

    glGenBuffers(1, &vbo);
    glBindBuffer(GL_ARRAY_BUFFER, vbo);
    glBufferData(GL_ARRAY_BUFFER, scene.vertices.size() * sizeof(scene.vertices[0]), scene.vertices.data(), GL_STATIC_DRAW);

    glGenBuffers(1, &ebo);
    glBindBuffer(GL_ELEMENT_ARRAY_BUFFER, ebo);
    glBufferData(GL_ELEMENT_ARRAY_BUFFER, all_indices.size() * sizeof(all_indices[0]), all_indices.data(), GL_STATIC_DRAW);

    glEnableVertexAttribArray(0);
    glVertexAttribPointer(0, 3, GL_FLOAT, GL_FALSE, sizeof(scene_data::vertex), (void *)(0));
    glEnableVertexAttribArray(1);
    glVertexAttribPointer(1, 3, GL_FLOAT, GL_FALSE, sizeof(scene_data::vertex), (void *)(12));
    glEnableVertexAttribArray(0);
    glVertexAttribPointer(0, 3, GL_FLOAT, GL_FALSE, sizeof(scene_data::vertex), nullptr);
    glEnableVertexAttribArray(1);
    glVertexAttribPointer(1, 3, GL_FLOAT, GL_FALSE, sizeof(scene_data::vertex),
        (void *)offsetof(scene_data::vertex, normal));
    glEnableVertexAttribArray(2);
    glVertexAttribPointer(2, 2, GL_FLOAT, GL_FALSE, sizeof(scene_data::vertex),
        (void *)offsetof(scene_data::vertex, texcoord));

    GLuint debug_vao;
    glGenVertexArrays(1, &debug_vao);

    GLsizei shadow_map_hd = 2048;

    GLuint shadow_map;
    glGenTextures(1, &shadow_map);
    glBindTexture(GL_TEXTURE_2D, shadow_map);
    glTexParameterf(GL_TEXTURE_2D, GL_TEXTURE_MIN_FILTER, GL_LINEAR);
    glTexParameterf(GL_TEXTURE_2D, GL_TEXTURE_MAG_FILTER, GL_LINEAR);
    glTexParameterf(GL_TEXTURE_2D, GL_TEXTURE_WRAP_S, GL_CLAMP_TO_EDGE);
    glTexParameterf(GL_TEXTURE_2D, GL_TEXTURE_WRAP_T, GL_CLAMP_TO_EDGE);
    glTexImage2D(GL_TEXTURE_2D, 0, GL_RG32F,
        shadow_map_hd, shadow_map_hd, 0,
        GL_RGBA, GL_FLOAT, nullptr);

    GLuint renderbuffer;
    glGenRenderbuffers(1, &renderbuffer);
    glBindRenderbuffer(GL_RENDERBUFFER, renderbuffer);
    glRenderbufferStorage(GL_RENDERBUFFER, GL_DEPTH_COMPONENT24, shadow_map_hd, shadow_map_hd);

    GLuint shadow_fbo;
    glGenFramebuffers(1, &shadow_fbo);
    glBindFramebuffer(GL_DRAW_FRAMEBUFFER, shadow_fbo);
    glFramebufferTexture(GL_DRAW_FRAMEBUFFER, GL_COLOR_ATTACHMENT0, shadow_map, 0);
    glFramebufferRenderbuffer(GL_DRAW_FRAMEBUFFER, GL_DEPTH_ATTACHMENT, GL_RENDERBUFFER, renderbuffer);
    if (glCheckFramebufferStatus(GL_DRAW_FRAMEBUFFER) != GL_FRAMEBUFFER_COMPLETE)
        throw std::runtime_error("Incomplete framebuffer!");
    glBindFramebuffer(GL_DRAW_FRAMEBUFFER, 0);

    std::map<std::string, GLuint> albedo_tex, alpha_tex;

    auto load_tex = [](tex_data::image_data &tex) {
        glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MIN_FILTER, GL_LINEAR_MIPMAP_LINEAR);
        glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MAG_FILTER, GL_LINEAR);

        glTexImage2D(GL_TEXTURE_2D, 0, GL_RGBA8, tex.x, tex.y, 0,
            GL_RGBA, GL_UNSIGNED_BYTE, tex.data);
        glGenerateMipmap(GL_TEXTURE_2D);
        stbi_image_free(tex.data);
    };

    for (const auto &elem : scene.tex_dict) {
        GLuint texture;
        if (elem.second->albedo.exists) {
            glGenTextures(1, &texture);
            albedo_tex[elem.first] = texture;
            glBindTexture(GL_TEXTURE_2D, texture);
            load_tex(elem.second->albedo);
        }
        if (elem.second->alpha.exists) {
            glGenTextures(1, &texture);
            alpha_tex[elem.first] = texture;
            glBindTexture(GL_TEXTURE_2D, texture);
            load_tex(elem.second->alpha);
        }
    }

    auto last_frame_start = std::chrono::high_resolution_clock::now();

    float time = 0.f;

    std::map<SDL_Keycode, bool> button_down;

    auto rotate_xz = [](glm::vec3 &vec, float angle) {
        glm::mat4 rotationMatrix = glm::rotate(glm::mat4(1.0f), angle, glm::vec3(0.0f, 1.0f, 0.0f));
        return (rotationMatrix * glm::vec4(vec, 1.0f)).xyz();
    };

    float view_dy = 0.f;
    float view_dx = 0.f;
    glm::vec3 dist(x_mid, y_mid, z_mid);
    glm::vec3 light(dist);
    glm::vec3 light_intensive(0.000001f, 0.009f, 0.00000005f);
    glm::vec3 light_shift(0.f);
    glm::vec3 camera_position(dist);
    glm::vec3 step(x_w / 15.f, 0, 0);
    bool running = true;
    while (running)
    {
        for (SDL_Event event; SDL_PollEvent(&event);) switch (event.type)
        {
        case SDL_QUIT:
            running = false;
            break;
        case SDL_WINDOWEVENT: switch (event.window.event)
        {
        case SDL_WINDOWEVENT_RESIZED:
            width = event.window.data1;
            height = event.window.data2;
            glViewport(0, 0, width, height);
            break;
        }
                            break;
        case SDL_KEYDOWN:
            button_down[event.key.keysym.sym] = true;
            break;
        case SDL_KEYUP:
            button_down[event.key.keysym.sym] = false;
            break;
        case SDL_MOUSEBUTTONDOWN:
            button_down[event.button.button] = true;
            break;
        case SDL_MOUSEBUTTONUP:
            button_down[event.button.button] = false;
            break;
        case SDL_MOUSEWHEEL:
            light_intensive.z -= event.wheel.y / 100000.f;
            light_intensive.z = std::max(light_intensive.z, 0.f);
            break;
        case SDL_MOUSEMOTION:
            int x, y;
            SDL_GetMouseState(&x, &y);

            auto [dx, dy] = std::pair{ (width / 100000.f) * (x - center_x), (height / 100000.f) * (y - center_y) };

            view_dx += dx;
            view_dy += dy;

            step = rotate_xz(step, -dx);

            SDL_WarpMouseInWindow(window, center_x, center_y);
            break;
        }

        if (!running)
            break;

        auto now = std::chrono::high_resolution_clock::now();
        float dt = std::chrono::duration_cast<std::chrono::duration<float>>(now - last_frame_start).count();
        last_frame_start = now;

        time += dt;

        if (button_down[SDLK_f] || button_down[SDL_BUTTON_LEFT]) {
            light = camera_position;
            light_shift = glm::vec3(0.f);
        }

        if (button_down[SDLK_t]) {
            light_intensive.y -= dt / 400.f;
            light_intensive.y = std::max(light_intensive.y, 0.f);
        }
        if (button_down[SDLK_g]) {
            light_intensive.y += dt / 400.f;
            light_intensive.y = std::max(light_intensive.y, 0.f);
        }

        if (button_down[SDLK_w])
            dist -= rotate_xz(step, -glm::pi<float>() / 2.f) * dt;
        if (button_down[SDLK_s])
            dist += rotate_xz(step, -glm::pi<float>() / 2.f) * dt;
        if (button_down[SDLK_a])
            dist -= step * dt;
        if (button_down[SDLK_d])
            dist += step * dt;

        if (button_down[SDLK_UP])
            light_shift -= rotate_xz(step, -glm::pi<float>() / 2.f) * dt;
        if (button_down[SDLK_DOWN])
            light_shift += rotate_xz(step, -glm::pi<float>() / 2.f) * dt;
        if (button_down[SDLK_LEFT])
            light_shift -= step * dt;
        if (button_down[SDLK_RIGHT])
            light_shift += step * dt;
        if (button_down[SDLK_i])
            light_shift.y += y_w / 15.f * dt;
        if (button_down[SDLK_k])
            light_shift.y -= y_w / 15.f * dt;

        if (button_down[SDLK_SPACE])
            dist.y += y_w / 6.f * dt;
        if (button_down[SDLK_LSHIFT] || button_down[SDLK_LCTRL])
            dist.y -= y_w / 6.f * dt;

        if (button_down[SDLK_ESCAPE])
            running = false;



        glm::mat4 model(1.f);
        glm::vec3 sun_direction = glm::normalize(glm::vec3(std::cos(time * 0.12f), 1.f, std::sin(time * 0.12f)));

        glBindFramebuffer(GL_DRAW_FRAMEBUFFER, shadow_fbo);
        glClearColor(1, 1, 0, 0);
        glClear(GL_COLOR_BUFFER_BIT | GL_DEPTH_BUFFER_BIT);
        glViewport(0, 0, shadow_map_hd, shadow_map_hd);

        glEnable(GL_DEPTH_TEST);
        glDepthFunc(GL_LEQUAL);

        glEnable(GL_CULL_FACE);
        glCullFace(GL_BACK);

        glm::vec3 light_z = -sun_direction;
        glm::vec3 light_x = glm::normalize(glm::cross(light_z, { 0.f, 1.f, 0.f }));
        glm::vec3 light_y = glm::cross(light_x, light_z);

        float max_len_x = 0, max_len_y = 0, max_len_z = 0;
        for (int i = 0; i < 8; ++i) {
            glm::vec3 v_c = { pow(-1, i) * x_w / 2, pow(-1, i / 2) * y_w / 2, pow(-1, i / 4) * z_w / 2 };
            max_len_x = fmax(max_len_x, abs(glm::dot(v_c, light_x)));
            max_len_y = fmax(max_len_y, abs(glm::dot(v_c, light_y)));
            max_len_z = fmax(max_len_z, abs(glm::dot(v_c, light_z)));
        }
        light_x *= max_len_x;
        light_y *= max_len_y;
        light_z *= max_len_z;

        glm::mat4 transform = {
                light_x.x, light_y.x, light_z.x, x_mid,
                light_x.y, light_y.y, light_z.y, y_mid,
                light_x.z, light_y.z, light_z.z, z_mid,
                0.0,             0.0,       0.0, 1.0,
        };
        transform = glm::transpose(transform);
        transform = glm::inverse(transform);

        glUseProgram(shadow_program);

        glUniformMatrix4fv(shadow_model_location, 1, GL_FALSE, reinterpret_cast<float *>(&model));
        glUniformMatrix4fv(shadow_transform_location, 1, GL_FALSE, reinterpret_cast<float *>(&transform));

        glBindVertexArray(vao);
        for (int i = 0; i < object_num; ++i) {
            auto object = *(scene.objects)[i];

            if (object.texture_added && scene.tex_dict[object.tex_name]->alpha.exists) {
                glActiveTexture(GL_TEXTURE0 + 1);
                glBindTexture(GL_TEXTURE_2D, alpha_tex[object.tex_name]);
                glUniform1i(shadow_is_alpha_location, 1);
                glUniform1i(shadow_alpha_texture_location, 1);
            }
            else
                glUniform1i(shadow_is_alpha_location, 0);

            glDrawElements(GL_TRIANGLES, draw_idx[i].second, GL_UNSIGNED_INT,
                (void *)(4 * draw_idx[i].first));
        }
        glActiveTexture(GL_TEXTURE0);
        glBindTexture(GL_TEXTURE_2D, shadow_map);
        glGenerateMipmap(GL_TEXTURE_2D);

        glBindFramebuffer(GL_DRAW_FRAMEBUFFER, 0);
        glViewport(0, 0, width, height);

        glClearColor(0.9f, 0.8f, 0.7f, 0.f);
        glClear(GL_COLOR_BUFFER_BIT | GL_DEPTH_BUFFER_BIT);

        glEnable(GL_DEPTH_TEST);
        glDepthFunc(GL_LEQUAL);

        glEnable(GL_CULL_FACE);
        glCullFace(GL_BACK);

        float near = 0.01f;
        float far = 5000.f;

        glm::mat4 view(1.f);
        view = glm::rotate(view, view_dy, { 1.f, 0.f, 0.f });
        view = glm::rotate(view, view_dx, { 0.f, 1.f, 0.f });
        view = glm::translate(view, { -dist.x, -dist.y, -dist.z });

        camera_position = (glm::inverse(view) * glm::vec4(0.f, 0.f, 0.f, 1.f)).xyz();

        glm::mat4 projection = glm::mat4(1.f);
        projection = glm::perspective(glm::pi<float>() / 2.f, (1.f * width) / height, near, far);

        glBindTexture(GL_TEXTURE_2D, shadow_map);

        glUseProgram(program);
        glUniform1i(shadow_map_location, 0);
        glUniformMatrix4fv(model_location, 1, GL_FALSE, reinterpret_cast<float *>(&model));
        glUniformMatrix4fv(view_location, 1, GL_FALSE, reinterpret_cast<float *>(&view));
        glUniformMatrix4fv(projection_location, 1, GL_FALSE, reinterpret_cast<float *>(&projection));
        glUniformMatrix4fv(transform_location, 1, GL_FALSE, reinterpret_cast<float *>(&transform));

        glUniform3f(ambient_location, 0.2f, 0.2f, 0.2f);
        glUniform3fv(sun_direction_location, 1, reinterpret_cast<float *>(&sun_direction));
        glUniform3f(sun_color_location, 0.9f, 0.8f, 0.4f);
        auto finaly_light = light + light_shift;
        auto camera_dir = camera_position + 5.f * step;

        float near_plane = 1.0f, far_plane = 7.5f;
        glm::mat4 lightProjection = glm::ortho(x_min, x_max, z_min, z_max, near, far);
        glm::mat4 lightView = glm::lookAt(glm::vec3(camera_position.x, camera_position.y, camera_position.z),
            glm::vec3(1.f, 0.f, 0.f),
            glm::vec3(0.0f, 1.0f, 0.0f));
        
        glm::mat4 lightSpaceMatrix = lightProjection * lightView;
        glUniformMatrix4fv(lightSpaceMatrix_location, 1, GL_FALSE, reinterpret_cast<float *>(&lightSpaceMatrix));
        glUniform3f(pl_position_location, finaly_light.x, finaly_light.y, finaly_light.z);
        glUniform3f(pl_color_location, 0.5f, 0.4f, 0.8f);
        glUniform3f(pl_attenuation_location, light_intensive.x, light_intensive.y, light_intensive.z);

        glUniform3fv(camera_location, 1, reinterpret_cast<float *>(&camera_position));

        glBindVertexArray(vao);
        for (int i = 0; i < object_num; ++i) {
            auto object = *(scene.objects)[i];

            if (object.texture_added && scene.tex_dict[object.tex_name]->albedo.exists) {
                glActiveTexture(GL_TEXTURE0 + 1);
                glBindTexture(GL_TEXTURE_2D, albedo_tex[object.tex_name]);
                glUniform1i(is_albedo_location, 1);
                glUniform1i(albedo_texture_location, 1);
            }
            else
                glUniform1i(is_albedo_location, 0);

            if (object.texture_added && scene.tex_dict[object.tex_name]->alpha.exists) {
                glActiveTexture(GL_TEXTURE0 + 2);
                glBindTexture(GL_TEXTURE_2D, alpha_tex[object.tex_name]);
                glUniform1i(is_alpha_location, 1);
                glUniform1i(alpha_texture_location, 2);
            }
            else
                glUniform1i(is_alpha_location, 0);

            if (object.texture_added) {
                glUniform3f(glossiness_location, scene.tex_dict[object.tex_name]->glossiness[0],
                    scene.tex_dict[object.tex_name]->glossiness[1],
                    scene.tex_dict[object.tex_name]->glossiness[2]);
                glUniform1f(roughness_location, scene.tex_dict[object.tex_name]->power);
            }
            else {
                glUniform3f(glossiness_location, 0.f, 0.f, 0.f);
                glUniform1f(roughness_location, 0.f);
            }

            glDrawElements(GL_TRIANGLES, draw_idx[i].second, GL_UNSIGNED_INT,
                (void *)(4 * draw_idx[i].first));
        }

        SDL_GL_SwapWindow(window);
    }

    SDL_GL_DeleteContext(gl_context);
    SDL_DestroyWindow(window);
}
catch (std::exception const &e)
{
    std::cerr << e.what() << std::endl;
    return EXIT_FAILURE;
}