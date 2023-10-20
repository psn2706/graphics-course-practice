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
#include <unordered_map>
#include <vector>


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
    throw std::runtime_error(to_string(message) +
        reinterpret_cast<const char *>(glewGetErrorString(error)));
}

const char vertex_shader_source_colored[] =
R"(
#version 330 core

uniform mat4 view;

layout (location = 0) in vec2 position;
layout (location = 1) in vec4 in_Color;

out vec4 color;

void main()
{
    gl_Position = view * vec4(position, 0.0, 1.0);
    color = in_Color;	
}
)";

const char vertex_shader_source[] =
R"(#version 330 core

uniform mat4 view;

layout (location = 0) in vec2 position;

out vec4 color;

void main()
{
    gl_Position = view * vec4(position, 0.0, 1.0);
    color = vec4(0.0, 0.0, 0.0, 1.0);
}
)";

const char fragment_shader_source[] =
R"(#version 330 core

in vec4 color;

layout (location = 0) out vec4 out_color;

void main()
{
    out_color = color;
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

struct vec {
    float x, y;
};

struct clr {
    std::uint8_t r, g, b, t;
};

struct vertex {
    vec position;
    clr color;
};

int main() try
{
    if (SDL_Init(SDL_INIT_VIDEO) != 0)
        sdl2_fail("SDL_Init: ");

    SDL_GL_SetAttribute(SDL_GL_CONTEXT_MAJOR_VERSION, 3);
    SDL_GL_SetAttribute(SDL_GL_CONTEXT_MINOR_VERSION, 3);
    SDL_GL_SetAttribute(SDL_GL_CONTEXT_PROFILE_MASK, SDL_GL_CONTEXT_PROFILE_CORE);
    SDL_GL_SetAttribute(SDL_GL_DOUBLEBUFFER, 1);
    SDL_GL_SetAttribute(SDL_GL_MULTISAMPLEBUFFERS, 1);
    SDL_GL_SetAttribute(SDL_GL_MULTISAMPLESAMPLES, 4);

    SDL_Window *window = SDL_CreateWindow("Graphics course practice 1",
        SDL_WINDOWPOS_CENTERED,
        SDL_WINDOWPOS_CENTERED,
        800, 600,
        SDL_WINDOW_OPENGL | SDL_WINDOW_RESIZABLE | SDL_WINDOW_MAXIMIZED);

    if (!window)
        sdl2_fail("SDL_CreateWindow: ");

    int width, height;
    SDL_GetWindowSize(window, &width, &height);

    SDL_GLContext gl_context = SDL_GL_CreateContext(window);
    if (!gl_context)
        sdl2_fail("SDL_GL_CreateContext: ");

    SDL_GL_SetSwapInterval(0);

    if (auto result = glewInit(); result != GLEW_NO_ERROR)
        glew_fail("glewInit: ", result);

    if (!GLEW_VERSION_3_3)
        throw std::runtime_error("OpenGL 3.3 is not supported");

    glClearColor(0.8f, 0.8f, 1.f, 0.f);

    // preparing

    GLuint vertex_shader_colored = create_shader(GL_VERTEX_SHADER, vertex_shader_source_colored);
    GLuint vertex_shader = create_shader(GL_VERTEX_SHADER, vertex_shader_source);
    GLuint fragment_shader = create_shader(GL_FRAGMENT_SHADER, fragment_shader_source);
    GLuint program = create_program(vertex_shader_colored, fragment_shader);
    GLuint program_isolines = create_program(vertex_shader, fragment_shader);
    GLuint view_location = glGetUniformLocation(program, "view");
    GLuint view_location_isolines = glGetUniformLocation(program_isolines, "view");

    GLuint vao, vind, vao_isolines, vind_isolines;
    GLuint vbo_position, vbo_color, vbo_isolines;
    glGenVertexArrays(1, &vao);
    glGenBuffers(1, &vbo_position);
    glGenBuffers(1, &vbo_color);
    glGenBuffers(1, &vind);
    glGenVertexArrays(1, &vao_isolines);
    glGenBuffers(1, &vbo_isolines);
    glGenBuffers(1, &vind_isolines);


    glBindVertexArray(vao);
    glBindBuffer(GL_ARRAY_BUFFER, vbo_position);
    glEnableVertexAttribArray(0);
    glVertexAttribPointer(0, 2, GL_FLOAT, GL_FALSE, sizeof(vertex), (void *)(0));

    glBindVertexArray(vao);
    glBindBuffer(GL_ARRAY_BUFFER, vbo_color);
    glEnableVertexAttribArray(1);
    glVertexAttribPointer(1, 4, GL_UNSIGNED_BYTE, GL_TRUE, sizeof(vertex),
        (void *)(offsetof(vertex, color)));


    glBindVertexArray(vao_isolines);
    glBindBuffer(GL_ARRAY_BUFFER, vbo_isolines);
    glEnableVertexAttribArray(0);
    glVertexAttribPointer(0, 2, GL_FLOAT, GL_FALSE, sizeof(vec), (void *)(0));



    float view[16] = {
        1.f, 0.f, 0.f, 0.f,
        0.f, 1.f, 0.f, 0.f,
        0.f, 0.f, 1.f, 0.f,
        0.f, 0.f, 0.f, 1.f
    };

    std::vector<vertex> grid;
    std::vector<std::uint32_t> indexes;
    float square = 20;
    float precision_speed = 150;
    std::uint32_t gridWidth, gridHeight;

    auto rebuild = [&]() {
        gridWidth = width / square + 2;
        gridHeight = height / square + 2;

        grid.resize(gridWidth * gridHeight);
        for (std::uint32_t i = 0; i < gridHeight; ++i) {
            for (std::uint32_t j = 0; j < gridWidth; ++j) {
                auto &cur = grid[i * gridWidth + j];

                cur.position.x = j * square;
                cur.position.y = i * square;
            }
        }
        indexes.resize((gridWidth - 1) * (gridHeight - 1) * 6);
        for (std::uint32_t i = 0; i < gridHeight - 1; ++i) {
            for (std::uint32_t j = 0; j < gridWidth - 1; ++j) {
                std::uint32_t cell = i * gridWidth + j;
                std::size_t cur = (i * (gridWidth - 1) + j) * 6;

                indexes[cur] = cell;
                indexes[cur + 1] = cell + 1;
                indexes[cur + 2] = cell + gridWidth;

                indexes[cur + 3] = cell + gridWidth;
                indexes[cur + 4] = cell + gridWidth + 1;
                indexes[cur + 5] = cell + 1;
            }
        }
        glBindVertexArray(vao);
        glBindBuffer(GL_ARRAY_BUFFER, vbo_position);
        glBufferData(GL_ARRAY_BUFFER, GLsizeiptr(grid.size() * sizeof(vertex)), grid.data(), GL_DYNAMIC_DRAW);
        glBindBuffer(GL_ELEMENT_ARRAY_BUFFER, vind);
        glBufferData(GL_ELEMENT_ARRAY_BUFFER, GLsizeiptr(indexes.size() * sizeof(std::uint32_t)), indexes.data(), GL_DYNAMIC_DRAW);
    };

    std::vector<vec> points;
    std::vector<std::uint32_t> pindexes;
    int count_of_isolines = 5;
    std::vector<float> isolines;

    auto build_isolines = [&]() {
        isolines.clear();
        float step = 2.f / (count_of_isolines + 1);
        for (int i = 0; i < count_of_isolines; ++i) {
            isolines.push_back(-1 + (i + 1) * step);
        }
    };

    build_isolines();

    // preparing

    auto last_frame_start = std::chrono::high_resolution_clock::now();

    float time = 0.f;
    float timeMultiplier = 1.0;
    int type = 3;
    std::unordered_map<SDL_Scancode, bool> key_down;

    // color calculator


    auto func = [&time, &timeMultiplier, &type](float x, float y) {
        float t = time * timeMultiplier;
        if (type == 1)
            return 1 / (2 + std::sqrt(x * x + y * y) / (500 + 400 * std::sin(t))) * 4 - 1;
        if (type == 2)
            return std::cos(std::sqrt(x * x + y * y) + t);
        if (type == 3)
            return (std::sin(x + t) + std::cos(y - t)) / 2;
    };



    // color calculator

    auto color = [](float z) -> clr {
        float t = (z + 1) / 2;
        clr plus = { 238, 144, 134, 255 };
        clr minus = { 135, 206, 235, 255 };
        return {
            std::uint8_t(plus.r * t + minus.r * (1 - t)),
            std::uint8_t(plus.g * t + minus.g * (1 - t)),
            std::uint8_t(plus.b * t + minus.b * (1 - t)),
            255 };
    };

    auto interpolate = [](vec l, vec r, float t) -> vec {
        return {
            l.x * (1 - t) + r.x * t,
            l.y * (1 - t) + r.y * t
        };
    };

    bool time_stop = false;
    bool grid_mode = false;

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

            rebuild();

            view[0] = 2.0 / width;
            view[3] = -1.f;
            view[5] = -2.0 / height;
            view[7] = 1.f;

            break;
        }
                            break;
        case SDL_KEYDOWN:
            key_down[event.key.keysym.scancode] = true;
            break;
        case SDL_KEYUP:
            key_down[event.key.keysym.scancode] = false;
            switch (event.key.keysym.scancode) {
            case SDL_SCANCODE_1:
                type = 1;
                rebuild();
                break;
            case SDL_SCANCODE_2:
                type = 2;
                rebuild();
                break;
            case SDL_SCANCODE_3:
                type = 3;
                rebuild();
                break;
            case SDL_SCANCODE_SPACE:
                time_stop = !time_stop;
                break;
            case SDL_SCANCODE_G:
                grid_mode = !grid_mode;
                break;
            case SDL_SCANCODE_KP_PLUS:
                count_of_isolines = std::min(count_of_isolines + 1, 10);
                build_isolines();
                break;
            case SDL_SCANCODE_KP_MINUS:
                count_of_isolines = std::max(count_of_isolines - 1, 1);
                build_isolines();
                break;
            }
            break;
        }

        if (!running)
            break;

        auto now = std::chrono::high_resolution_clock::now();
        float dt = std::chrono::duration_cast<std::chrono::duration<float>>(now - last_frame_start).count();
        last_frame_start = now;
        if (!time_stop) {
            time += dt;
        }

        if (key_down[SDL_SCANCODE_LEFT]) {
            timeMultiplier = std::max(timeMultiplier - 0.01f, 0.1f);
            //rebuild();
        }
        if (key_down[SDL_SCANCODE_RIGHT]) {
            timeMultiplier = std::min(timeMultiplier + 0.01f, 5.f);
            //rebuild();
        }
        if (key_down[SDL_SCANCODE_UP]) {
            square = std::min(square + precision_speed * dt, 500.f);
            rebuild();
        }
        if (key_down[SDL_SCANCODE_DOWN]) {
            square = std::max(square - precision_speed * dt, 10.f);
            rebuild();
        }

        // updating

        for (std::uint32_t i = 0; i < gridHeight; ++i) {
            for (std::uint32_t j = 0; j < gridWidth; ++j) {
                auto &cur = grid[i * gridWidth + j];

                cur.color = color(func(cur.position.x, cur.position.y));
            }
        }
        glBindVertexArray(vao);
        glBindBuffer(GL_ARRAY_BUFFER, vbo_color);
        glBufferData(GL_ARRAY_BUFFER, GLsizeiptr(grid.size() * sizeof(vertex)), grid.data(), GL_DYNAMIC_DRAW);

        /// isolines

        points.clear();
        pindexes.clear();

        for (auto Const : isolines) {
            for (int i = 0; i < gridHeight - 1; ++i) {
                for (int j = 0; j < gridWidth - 1; ++j) {
                    // a -- b
                    // |    |
                    // c -- d
                    vec pa = grid[i * gridWidth + j].position,
                        pb = grid[i * gridWidth + j + 1].position,
                        pc = grid[(i + 1) * gridWidth + j].position,
                        pd = grid[(i + 1) * gridWidth + j + 1].position;
                    float a = func(pa.x, pa.y),
                        b = func(pb.x, pb.y),
                        c = func(pc.x, pc.y),
                        d = func(pd.x, pd.y);
                    bool sa = a > Const, sb = b > Const, sc = c > Const, sd = d > Const;
                    if (sb ^ sc) {
                        pindexes.push_back(points.size());
                        if (sa ^ sb) {
                            points.push_back(interpolate(pa, pb, (Const - a) / (b - a)));
                        }
                        else {
                            points.push_back(interpolate(pa, pc, (Const - a) / (c - a)));
                        }
                        pindexes.push_back(points.size());
                        pindexes.push_back(points.size());
                        points.push_back(interpolate(pb, pc, (Const - b) / (c - b)));

                        pindexes.push_back(points.size());
                        if (sd ^ sb) {
                            points.push_back(interpolate(pd, pb, (Const - d) / (b - d)));
                        }
                        else {
                            points.push_back(interpolate(pc, pd, (Const - c) / (d - c)));
                        }
                    }
                    else {
                        if (sa ^ sb) {
                            pindexes.push_back(points.size());
                            points.push_back(interpolate(pa, pb, (Const - a) / (b - a)));
                            pindexes.push_back(points.size());
                            points.push_back(interpolate(pa, pc, (Const - a) / (c - a)));
                        }
                        if (sd ^ sb) {
                            pindexes.push_back(points.size());
                            points.push_back(interpolate(pd, pb, (Const - d) / (b - d)));
                            pindexes.push_back(points.size());
                            points.push_back(interpolate(pd, pc, (Const - d) / (c - d)));
                        }
                    }
                }
            }
        }

        glBindVertexArray(vao_isolines);
        glBindBuffer(GL_ARRAY_BUFFER, vbo_isolines);
        glBufferData(GL_ARRAY_BUFFER, GLsizeiptr(points.size() * sizeof(vec)), points.data(), GL_DYNAMIC_DRAW);
        glBindBuffer(GL_ELEMENT_ARRAY_BUFFER, vind_isolines);
        glBufferData(GL_ELEMENT_ARRAY_BUFFER, GLsizeiptr(pindexes.size() * sizeof(std::uint32_t)), pindexes.data(), GL_DYNAMIC_DRAW);

        /// isolines


        // updating

        glClear(GL_COLOR_BUFFER_BIT);

        // drawing

        glUseProgram(program);
        glUniformMatrix4fv(view_location, 1, GL_TRUE, view);
        glBindVertexArray(vao);
        glBindBuffer(GL_ELEMENT_ARRAY_BUFFER, vind);
        glDrawElements(grid_mode ? GL_LINES : GL_TRIANGLES, indexes.size(), GL_UNSIGNED_INT, nullptr);

        glUseProgram(program_isolines);
        glUniformMatrix4fv(view_location_isolines, 1, GL_TRUE, view);
        glBindVertexArray(vao_isolines);
        glBindBuffer(GL_ELEMENT_ARRAY_BUFFER, vind_isolines);
        glDrawElements(GL_LINES, pindexes.size(), GL_UNSIGNED_INT, nullptr);

        // drawing 
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