#define SDL_MAIN_HANDLED
#include <SDL2/SDL.h>

#include <cstdio>
#include <stdexcept>
#include <memory>
#include <array>
#include <cmath>

#define FPS 30
#define FRAME_TARGET_TIME (1000 / FPS)

#define MESH_VERTICES 8
#define MESH_FACES (6 * 2) /* 6 cube face and 2 triangles per face */

struct Vec2
{
    float x;
    float y;
};

struct Vec3
{
    float x;
    float y;
    float z;
};

/* It stores the indices */
struct Face
{
    int a;
    int b;
    int c;
};

struct Triangle
{
    std::array<Vec2, 3> points;
};

Vec3 vec3_rotate_x(const Vec3& v, float angle)
{
    Vec3 rotated_vector = {
        .x = v.x,
        .y = v.y * std::cos(angle) - v.z * std::sin(angle),
        .z = v.y * std::sin(angle) + v.z * std::cos(angle)
    };
    return rotated_vector;
}

Vec3 vec3_rotate_y(const Vec3& v, float angle)
{
    Vec3 rotated_vector = {
        .x = v.x * std::cos(angle) - v.z * std::sin(angle),
        .y = v.y,
        .z = v.x * std::sin(angle) + v.z * std::cos(angle)
    };
    return rotated_vector;
}

Vec3 vec3_rotate_z(const Vec3& v, float angle)
{
    Vec3 rotated_vector = {
        .x = v.x * std::cos(angle) - v.y * std::sin(angle),
        .y = v.x * std::sin(angle) + v.y * std::cos(angle),
        .z = v.z
    };
    return rotated_vector;
}

template<uint32_t WIDTH, uint32_t HEIGHT>
class Engine
{
    /* If any of them is zero then do assert */
    static_assert(WIDTH != 0 || HEIGHT != 0, "Either WIDTH or HEIGHT must be non-zero");

    public:
        Engine()
        {}

        void Setup()
        {
            _colorBufferTexture = SDL_CreateTexture(_renderer, SDL_PIXELFORMAT_ARGB8888, SDL_TEXTUREACCESS_STREAMING, WIDTH, HEIGHT);

            int point_count = 0;
            for(float x = -1; x <= 1; x += 0.25)
            {
                for(float y = -1; y <= 1; y += 0.25)
                {
                    for(float z = -1; z <= 1; z += 0.25)
                    {
                        _cubePoints[point_count++] = {x,y,z};
                    }
                }
            }
        }

        void ProcessInput()
        {
            SDL_Event event;
            SDL_PollEvent(&event);

            switch(event.type)
            {
                case SDL_QUIT: /* When close button is pressed */
                    _isRunning = false;
                    break;

                case SDL_KEYDOWN:
                    if(event.key.keysym.sym == SDLK_ESCAPE)
                    {
                        /* When ESC button is pressed */
                        _isRunning = false;
                    }
                    break;
            }
        }

        void Update()
        {
            /* Wait some time until it reach the target frame time in milliseconds */
            int time_to_wait = FRAME_TARGET_TIME - (SDL_GetTicks() - _previous_frame_time);

            /* Only delay execution if we are running too fast */
            if(time_to_wait > 0 && time_to_wait <= FRAME_TARGET_TIME)
            {
                SDL_Delay(time_to_wait);
            }

            _previous_frame_time = SDL_GetTicks();

            _cubeRotation.x += 0.01;
            _cubeRotation.y += 0.01;
            _cubeRotation.z += 0.01;

            /* Loop all triangle faces of our mesh */
            for(int i = 0; i < MESH_FACES; i++)
            {   
                const Face& mesh_face = _mesh_faces[i];

                Vec3 face_vertices[3];
                face_vertices[0] = _mesh_vertices[mesh_face.a - 1];
                face_vertices[1] = _mesh_vertices[mesh_face.b - 1];
                face_vertices[2] = _mesh_vertices[mesh_face.c - 1];

                Triangle projected_triangle;
                
                /* Loop all three vertices of this current face and apply transformations */
                for(int j = 0; j < 3; j++)
                {
                    Vec3 transformed_vertex = face_vertices[j];

                    transformed_vertex = vec3_rotate_x(transformed_vertex, _cubeRotation.x);
                    transformed_vertex = vec3_rotate_y(transformed_vertex, _cubeRotation.y);
                    transformed_vertex = vec3_rotate_z(transformed_vertex, _cubeRotation.z);
                    
                    /* Translate the vertex away from the camera */
                    transformed_vertex.z -= _cameraPosition.z;

                    /* Project the current point */
                    Vec2 projected_point = Project(transformed_vertex);

                    /* Scale and translate the projected points to the middle of the screen */
                    projected_point.x += (WIDTH/2.0f);
                    projected_point.y += (HEIGHT/2.0f);

                    projected_triangle.points[j] = std::move(projected_point);
                }

                /* Save the projected triangle in the array of triangles to render */
                _triangles_to_render[i] = std::move(projected_triangle);
            }
        }

        void Render()
        {
            FillColorBuffer(0xFF000000);
            // DrawGrid();

            // Loop all projected points and render them
            for(int i = 0; i < MESH_FACES; i++)
            {
                const Triangle& triangle = _triangles_to_render[i];
                DrawRectangle({static_cast<uint32_t>(triangle.points[0].x), static_cast<uint32_t>(triangle.points[0].y), 3, 3, 0xFFFFFF00});
                DrawRectangle({static_cast<uint32_t>(triangle.points[1].x), static_cast<uint32_t>(triangle.points[1].y), 3, 3, 0xFFFFFF00});
                DrawRectangle({static_cast<uint32_t>(triangle.points[2].x), static_cast<uint32_t>(triangle.points[2].y), 3, 3, 0xFFFFFF00});
            }
            
            RenderColorBuffer();
            SDL_RenderPresent(_renderer);
        }

        void InitializeWindow()
        {
            if(SDL_Init(SDL_INIT_EVERYTHING) != 0)
            {
                throw std::runtime_error("Error initializing SDL");
            }

            _window = SDL_CreateWindow(nullptr, SDL_WINDOWPOS_CENTERED, SDL_WINDOWPOS_CENTERED, WIDTH, HEIGHT, SDL_WINDOW_BORDERLESS);
            if(!_window)
            {
                throw std::runtime_error("Error creating SDL Window");
            }

            _renderer = SDL_CreateRenderer(_window, -1, 0);
            if(!_renderer)
            {
                throw std::runtime_error("Error creating SDL Renderer");
            }

            _isRunning = true;
        }

        void DestroyWindow()
        {
            SDL_DestroyRenderer(_renderer);
            SDL_DestroyWindow(_window);
            SDL_Quit();
        }

        bool IsRunning() const
        {
            return _isRunning;
        }

    private:
        void DrawGrid()
            {
                for(int i = 0; i < WIDTH; i++)
                {
                    for(int j = 0; j < HEIGHT; j++)
                    {
                        if(i % 10 == 0 || j % 10 == 0)
                        {
                            DrawPixel(i, j, 0xFF333333);
                        }
                    }
                }
                
            }

        void DrawRectangle(const std::array<uint32_t, 5>& rectInfo)
        {
            /*  
                <x:0,y:1,w:2,h:3,color:4> :- 
            
                (x,y) is Position
                (w,h) is Size
                color is SDL_PIXELFORMAT_ARGB8888
            
            */
            for(int i = 0; i < rectInfo[2]; i++)
            {
                for(int j = 0; j < rectInfo[3]; j++)
                {
                    uint32_t current_x = rectInfo[0] + i;
                    uint32_t current_y = rectInfo[1] + j;
                    
                    DrawPixel(current_x, current_y, rectInfo[4]);
                }
            }

        }

        void FillColorBuffer(uint32_t color)
        {
            for(int i = 0; i < WIDTH; i++)
            {
                for(int j = 0; j < HEIGHT; j++)
                {
                    DrawPixel(i, j, color);
                }
            }
        }

        void DrawPixel(int x, int y, uint32_t color)
        {
            if(x >= 0 && x < WIDTH && y >= 0 && y < HEIGHT)
            {
                _colorBuffer[(WIDTH * y) + x] = color;
            }
        }

        /* Function that receives a 3D vector and returns a projected 2D point */
        Vec2 Project(const Vec3& point)
        {
            Vec2 projected_point = {
                ((_fovFactor * point.x) / point.z),
                ((_fovFactor * point.y) / point.z)
            };

            return projected_point;
        }

        void RenderColorBuffer()
        {
            SDL_UpdateTexture(_colorBufferTexture, nullptr, _colorBuffer.data(), WIDTH * sizeof(uint32_t));
            SDL_RenderCopy(_renderer, _colorBufferTexture, nullptr, nullptr);
        }

    private:
        std::array<uint32_t, WIDTH * HEIGHT> _colorBuffer;
        std::array<Vec3, 9*9*9> _cubePoints;
        std::array<Vec2, 9*9*9> _projectedCubePoints;

        bool _isRunning = false;
        uint32_t _fovFactor = 640;
        Vec3 _cameraPosition = {0, 0, -5};
        Vec3 _cubeRotation = {0, 0, 0};

        SDL_Window* _window;
        SDL_Renderer* _renderer;
        SDL_Texture* _colorBufferTexture;

        int _previous_frame_time;

        std::array<Triangle, MESH_FACES> _triangles_to_render;

        std::array<Vec3, MESH_VERTICES> _mesh_vertices = {
            Vec3{ -1.0f, -1.0f, -1.0f }, // 1
            Vec3{ -1.0f,  1.0f, -1.0f }, // 2
            Vec3{  1.0f,  1.0f, -1.0f }, // 3
            Vec3{  1.0f, -1.0f, -1.0f }, // 4
            Vec3{  1.0f,  1.0f,  1.0f }, // 5
            Vec3{  1.0f, -1.0f,  1.0f }, // 6
            Vec3{ -1.0f,  1.0f,  1.0f }, // 7
            Vec3{ -1.0f, -1.0f,  1.0f }  // 8
        };

        std::array<Face, MESH_FACES> _mesh_faces = {
            
            // front
            Face{ 1, 2, 3 },
            Face{ 1, 3, 4 },
            // right
            Face{ 4, 3, 5 },
            Face{ 4, 5, 6 },
            // back
            Face{ 6, 5, 7 },
            Face{ 6, 7, 8 },
            // left
            Face{ 8, 7, 2 },
            Face{ 8, 2, 1 },
            // top
            Face{ 2, 7, 5 },
            Face{ 2, 5, 3 },
            // bottom
            Face{ 6, 8, 1 },
            Face{ 6, 1, 4 }
        };
};

int main()
{   
    auto engine = std::make_unique<Engine<800, 600>>();
    try
    {
        engine->InitializeWindow();
        engine->Setup();

        while(engine->IsRunning())
        {
            engine->ProcessInput();
            engine->Update();
            engine->Render();
        }

        engine->DestroyWindow();
    }
    catch(const std::exception& e)
    {
        fprintf(stderr, "%s\n", e.what());
    }
    catch(...)
    {
        fprintf(stderr, "%s\n", "Unexpected Error\n");
    }
    return 0;
}
