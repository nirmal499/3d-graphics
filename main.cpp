#define SDL_MAIN_HANDLED
#include <SDL2/SDL.h>

#include <cstdio>
#include <stdexcept>
#include <memory>
#include <array>

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
            for(int i = 0; i < _cubePoints.size(); i++)
            {
                Vec3 point = _cubePoints[i];

                point.z -= _cameraPosition.z;

                // Project the current point 
                Vec2 projected_point = Project(point);

                // Save the projected 2D vector in the array of the projected points
                _projectedCubePoints[i] = std::move(projected_point);

            }
        }

        void Render()
        {
            FillColorBuffer(0xFF000000);
            // DrawGrid();

            // Loop all projected points and render them
            for(int i = 0; i < _projectedCubePoints.size(); i++)
            {
                const Vec2& projected_point = _projectedCubePoints[i];
                DrawRectangle({static_cast<uint32_t>(projected_point.x), static_cast<uint32_t>(projected_point.y), 4, 4, 0xFFFFFF00});
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
                ((_fovFactor * point.x) / point.z) + (WIDTH/2.0f) ,
                ((_fovFactor * point.y) / point.z) + (HEIGHT/2.0f)
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
        uint32_t _fovFactor = 128;
        Vec3 _cameraPosition = {0, 0, -2};

        SDL_Window* _window;
        SDL_Renderer* _renderer;
        SDL_Texture* _colorBufferTexture;
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
