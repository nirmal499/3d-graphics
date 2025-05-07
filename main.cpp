#define SDL_MAIN_HANDLED
#include <SDL2/SDL.h>

#include <cstdio>
#include <stdexcept>
#include <memory>
#include <array>

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
            /* Set the pixel at row 10 column 20 to color red */
            /* _colorBuffer[(WIDTH * 10) + 20] = 0xFFFF0000; */

            _colorBufferTexture = SDL_CreateTexture(_renderer, SDL_PIXELFORMAT_ARGB8888, SDL_TEXTUREACCESS_STREAMING, WIDTH, HEIGHT);

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

        }

        void Render()
        {
            SDL_SetRenderDrawColor(_renderer, 100, 130, 160, 255); /* (<R>, <G>, <B>, <0:Transparency::255:Opaque)> */
            SDL_RenderClear(_renderer); /* It clears/fills the renderer with the specified color */

            FillColorBuffer(0xFF000000);
            // DrawGrid();
            DrawRectangle({100, 150, 300, 150, 0xFFFF00FF});
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
            // for(int i = 0; i < WIDTH; i++)
            // {
            //     for(int j = 0; j < HEIGHT; j++)
            //     {
            //         if(i % 10 == 0 || j % 10 == 0)
            //         {
            //             // _colorBuffer[(WIDTH * y) + x] = 0xFF333333;
            //             DrawPixel(i, j, 0xFF333333);
            //         }
            //     }
            // }

            for(int i = 0; i < WIDTH; i++)
            {
                for(int j = 0; j < HEIGHT; j++)
                {
                    if(i % 10 == 0 && j % 10 == 0)
                    {
                        // _colorBuffer[(WIDTH * y) + x] = 0xFF333333;
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
                    
                    // _colorBuffer[(WIDTH * current_y) + current_x] = rectInfo[4];
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
                    // _colorBuffer[(WIDTH * y) + x] = color;
                    DrawPixel(i, j, color);
                }
            }
        }

        void DrawPixel(int x, int y, uint32_t color)
        {
            _colorBuffer[(WIDTH * y) + x] = color;
        }

        void RenderColorBuffer()
        {
            SDL_UpdateTexture(_colorBufferTexture, nullptr, _colorBuffer.data(), WIDTH * sizeof(uint32_t));
            SDL_RenderCopy(_renderer, _colorBufferTexture, nullptr, nullptr);
        }

    private:
        std::array<uint32_t, WIDTH * HEIGHT> _colorBuffer;
    
        bool _isRunning = false;

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
