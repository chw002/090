# AIWeatherArt Backend

A Flask-based microservices backend that integrates weather, news, and music services through a unified API gateway.

## Architecture

```mermaid
flowchart TB
    classDef external fill:#f9f9f9,stroke:#ddd,stroke-width:2px
    classDef service fill:#e1f5fe,stroke:#03a9f4,stroke-width:2px
    classDef main fill:#e8f5e9,stroke:#4caf50,stroke-width:2px
    classDef api fill:#fff3e0,stroke:#ff9800,stroke-width:2px

    Client[/"Client Applications"/]:::external
    
    subgraph MainApp["Main Application (Flask)"]:::main
        Gateway["API Gateway\n(DispatcherMiddleware)"]
        
        subgraph Services["Microservices"]
            Weather["Weather Service\n/weather"]:::service
            News["News Service\n/news"]:::service
            Music["Music Service\n/music"]:::service
        end
    end
    
    subgraph ExternalAPIs["External APIs"]:::external
        OpenWeather["OpenWeatherMap API\n(Weather Data)"]:::api
        GNews["GNews API\n(News Feed)"]:::api
        Spotify["Spotify API\n(Music Streaming)"]:::api
    end

    Client --> Gateway
    Gateway --> Weather & News & Music
    
    Weather -- "Current Weather\n& Forecast" --> OpenWeather
    News -- "Headlines\n& Categories" --> GNews
    Music -- "Search & Playback\nOAuth2 Auth" --> Spotify
```

## Features

### Weather Service (`/weather`)
- Real-time weather conditions
- 5-day weather forecasts
- Temperature, humidity, wind data
- Support for global cities

### News Service (`/news`)
- Latest news headlines by country
- Category filtering
- Custom search queries
- Multiple news sources

### Music Service (`/music`)
- Spotify integration
- Music search functionality
- Playback control
- OAuth2 authentication

## Setup

1. Clone the repository:
```bash
git clone https://github.com/yourusername/aiweatherart-backend.git
cd aiweatherart-backend
```

2. Install dependencies:
```bash
python setup.py
```

3. Configure environment variables:
   - Copy `.env.example` to `.env`
   - Add your API keys (see API Requirements section)

4. Run the application:
- Windows: `run.bat`
- Unix/Linux: `./run.sh`

## API Requirements

You'll need to obtain API keys from:
- [OpenWeatherMap](https://openweathermap.org/api)
- [GNews](https://gnews.io/)
- [Spotify Developer Dashboard](https://developer.spotify.com/dashboard/)

## Environment Variables

Required environment variables:
```env
WEATHER_API_KEY=your_openweathermap_api_key
GNEWS_API_KEY=your_gnews_api_key
SPOTIPY_CLIENT_ID=your_spotify_client_id
SPOTIPY_CLIENT_SECRET=your_spotify_client_secret
```

## Development

### Project Structure
```
.
├── src/
│   ├── main.py         # API Gateway & Main application
│   ├── weather.py      # Weather service implementation
│   ├── news.py         # News service implementation
│   └── music.py        # Music service implementation
├── test/
│   ├── test_weather.py # Weather service tests
│   ├── test_news.py    # News service tests
│   └── test_music.py   # Music service tests
├── requirements.txt    # Project dependencies
├── setup.py           # Installation script
└── run.bat/run.sh     # Startup scripts
```

### Testing

Run the test suite:
```bash
pytest test/
```

Individual service tests:
```bash
pytest test/test_weather.py
pytest test/test_news.py
pytest test/test_music.py
```

## API Documentation

### Weather Endpoints
- `GET /weather?city={city_name}` - Get current weather
- `GET /forecast?city={city_name}` - Get 5-day forecast

### News Endpoints
- `GET /news?country={country_code}&category={category}` - Get news headlines
- `GET /categories` - List available news categories

### Music Endpoints
- `GET /search?q={query}` - Search for tracks
- `GET /play` - Play selected track (requires authentication)

## Contributing

1. Fork the repository
2. Create your feature branch (`git checkout -b feature/AmazingFeature`)
3. Commit your changes (`git commit -m 'Add some AmazingFeature'`)
4. Push to the branch (`git push origin feature/AmazingFeature`)
5. Open a Pull Request

## License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.
