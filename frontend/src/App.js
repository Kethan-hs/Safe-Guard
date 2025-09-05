import { useState, useEffect } from "react";
import "./App.css";
import axios from "axios";
import { MapContainer, TileLayer, Marker, Popup, Circle, useMapEvents } from 'react-leaflet';
import { Card, CardContent, CardDescription, CardHeader, CardTitle } from "./components/ui/card";
import { Button } from "./components/ui/button";
import { Badge } from "./components/ui/badge";
import { Alert, AlertDescription } from "./components/ui/alert";
import { Tabs, TabsContent, TabsList, TabsTrigger } from "./components/ui/tabs";
import { Select, SelectContent, SelectItem, SelectTrigger, SelectValue } from "./components/ui/select";
import { Textarea } from "./components/ui/textarea";
import { Input } from "./components/ui/input";
import { Separator } from "./components/ui/separator";
import { Progress } from "./components/ui/progress";
import { 
  MapPin, 
  Shield, 
  AlertTriangle, 
  TrendingUp, 
  Users, 
  Clock,
  Target,
  BarChart3,
  Map,
  Database,
  AlertCircle,
  CheckCircle2,
  XCircle
} from "lucide-react";
import L from 'leaflet';

// Fix for default markers in react-leaflet
delete L.Icon.Default.prototype._getIconUrl;
L.Icon.Default.mergeOptions({
  iconRetinaUrl: 'https://cdnjs.cloudflare.com/ajax/libs/leaflet/1.7.1/images/marker-icon-2x.png',
  iconUrl: 'https://cdnjs.cloudflare.com/ajax/libs/leaflet/1.7.1/images/marker-icon.png',
  shadowUrl: 'https://cdnjs.cloudflare.com/ajax/libs/leaflet/1.7.1/images/marker-shadow.png',
});

const BACKEND_URL = process.env.REACT_APP_BACKEND_URL;
const API = `${BACKEND_URL}/api`;

// Interactive Map Component
const InteractiveMap = ({ onLocationSelect, crimeData, userLocation, safetyAnalysis }) => {
  const LocationMarker = () => {
    useMapEvents({
      click(e) {
        onLocationSelect(e.latlng);
      },
    });
    return null;
  };

  const getRiskColor = (severity) => {
    if (severity >= 8) return '#dc2626'; // red-600
    if (severity >= 6) return '#ea580c'; // orange-600
    if (severity >= 4) return '#ca8a04'; // yellow-600
    return '#16a34a'; // green-600
  };

  const getRiskColorFromLevel = (level) => {
    switch(level) {
      case 'critical': return '#dc2626';
      case 'high': return '#ea580c';
      case 'medium': return '#ca8a04';
      case 'low': return '#16a34a';
      default: return '#6b7280';
    }
  };

  return (
    <MapContainer
      center={[20.5937, 78.9629]} // Center of India
      zoom={5}
      style={{ height: '500px', width: '100%' }}
      className="rounded-lg border shadow-lg"
    >
      <TileLayer
        url="https://{s}.tile.openstreetmap.org/{z}/{x}/{y}.png"
        attribution='&copy; <a href="https://www.openstreetmap.org/copyright">OpenStreetMap</a> contributors'
      />
      <LocationMarker />
      
      {/* Crime markers */}
      {crimeData.map((crime, index) => (
        <Marker 
          key={index} 
          position={[crime.location.lat, crime.location.lng]}
          icon={L.divIcon({
            className: 'custom-crime-marker',
            html: `<div style="background-color: ${getRiskColor(crime.severity)}; width: 12px; height: 12px; border-radius: 50%; border: 2px solid white; box-shadow: 0 1px 3px rgba(0,0,0,0.3);"></div>`,
            iconSize: [12, 12],
            iconAnchor: [6, 6]
          })}
        >
          <Popup>
            <div className="p-2">
              <h4 className="font-semibold text-sm">{crime.crime_type.replace('_', ' ').toUpperCase()}</h4>
              <p className="text-xs text-gray-600">{crime.city}, {crime.state}</p>
              <p className="text-xs">Severity: {crime.severity}/10</p>
              <p className="text-xs">{crime.description}</p>
            </div>
          </Popup>
        </Marker>
      ))}
      
      {/* User location marker */}
      {userLocation && (
        <>
          <Marker 
            position={[userLocation.lat, userLocation.lng]}
            icon={L.divIcon({
              className: 'user-location-marker',
              html: '<div style="background-color: #3b82f6; width: 16px; height: 16px; border-radius: 50%; border: 3px solid white; box-shadow: 0 2px 6px rgba(0,0,0,0.3);"></div>',
              iconSize: [16, 16],
              iconAnchor: [8, 8]
            })}
          >
            <Popup>Your Location</Popup>
          </Marker>
          
          {/* Safety analysis circle */}
          {safetyAnalysis && (
            <Circle
              center={[userLocation.lat, userLocation.lng]}
              radius={5000} // 5km radius
              pathOptions={{
                color: getRiskColorFromLevel(safetyAnalysis.risk_level),
                weight: 2,
                opacity: 0.6,
                fillOpacity: 0.1
              }}
            />
          )}
        </>
      )}
    </MapContainer>
  );
};

// Safety Recommendations Component
const SafetyRecommendations = ({ analysis, prediction }) => {
  if (!analysis) return null;

  const getRiskBadgeVariant = (level) => {
    switch(level) {
      case 'critical': return 'destructive';
      case 'high': return 'destructive';
      case 'medium': return 'default';
      case 'low': return 'secondary';
      default: return 'outline';
    }
  };

  const getRiskIcon = (level) => {
    switch(level) {
      case 'critical': return <XCircle className="h-5 w-5" />;
      case 'high': return <AlertTriangle className="h-5 w-5" />;
      case 'medium': return <AlertCircle className="h-5 w-5" />;
      case 'low': return <CheckCircle2 className="h-5 w-5" />;
      default: return <Shield className="h-5 w-5" />;
    }
  };

  return (
    <div className="space-y-6">
      <Card>
        <CardHeader>
          <CardTitle className="flex items-center gap-2">
            <Shield className="h-5 w-5" />
            Location Safety Analysis
          </CardTitle>
        </CardHeader>
        <CardContent className="space-y-4">
          <div className="flex items-center justify-between">
            <div className="flex items-center gap-2">
              {getRiskIcon(analysis.risk_level)}
              <span className="text-lg font-semibold">Risk Level</span>
            </div>
            <Badge variant={getRiskBadgeVariant(analysis.risk_level)} className="text-sm">
              {analysis.risk_level.toUpperCase()}
            </Badge>
          </div>
          
          <div className="space-y-2">
            <div className="flex justify-between text-sm">
              <span>Risk Score</span>
              <span className="font-medium">{analysis.risk_score}/100</span>
            </div>
            <Progress value={analysis.risk_score} className="h-2" />
          </div>
          
          <Separator />
          
          <div className="space-y-3">
            <h4 className="font-semibold text-sm">Safety Recommendations</h4>
            <div className="space-y-2">
              {analysis.recommendations.map((rec, index) => (
                <Alert key={index} className="py-2">
                  <AlertDescription className="text-sm">{rec}</AlertDescription>
                </Alert>
              ))}
            </div>
          </div>
          
          {analysis.nearby_crimes.length > 0 && (
            <>
              <Separator />
              <div className="space-y-3">
                <h4 className="font-semibold text-sm">Recent Nearby Incidents ({analysis.nearby_crimes.length})</h4>
                <div className="space-y-2 max-h-40 overflow-y-auto">
                  {analysis.nearby_crimes.slice(0, 5).map((crime, index) => (
                    <div key={index} className="text-xs p-2 bg-gray-50 rounded-md">
                      <div className="flex justify-between items-start">
                        <span className="font-medium">{crime.crime_type.replace('_', ' ')}</span>
                        <Badge variant="outline" className="text-xs">
                          {crime.distance}km away
                        </Badge>
                      </div>
                      <div className="text-gray-600 mt-1">
                        Severity: {crime.severity}/10
                      </div>
                    </div>
                  ))}
                </div>
              </div>
            </>
          )}
        </CardContent>
      </Card>

      {prediction && (
        <Card>
          <CardHeader>
            <CardTitle className="flex items-center gap-2">
              <Target className="h-5 w-5" />
              AI Crime Prediction
            </CardTitle>
          </CardHeader>
          <CardContent className="space-y-4">
            <div className="space-y-2">
              <div className="flex justify-between text-sm">
                <span>Predicted Risk Score</span>
                <span className="font-medium">{prediction.predicted_risk_score}/100</span>
              </div>
              <Progress value={prediction.predicted_risk_score} className="h-2" />
            </div>
            
            <div className="space-y-2">
              <span className="text-sm font-medium">Likely Crime Types</span>
              <div className="flex flex-wrap gap-1">
                {prediction.predicted_crime_types.map((type, index) => (
                  <Badge key={index} variant="outline" className="text-xs">
                    {type.replace('_', ' ')}
                  </Badge>
                ))}
              </div>
            </div>
            
            <div className="text-sm text-gray-600">
              Confidence: {Math.round(prediction.confidence * 100)}%
            </div>
          </CardContent>
        </Card>
      )}
    </div>
  );
};

// Dashboard Stats Component
const DashboardStats = ({ stats }) => {
  if (!stats) return null;

  return (
    <div className="grid grid-cols-1 md:grid-cols-2 lg:grid-cols-4 gap-4 mb-6">
      <Card>
        <CardContent className="p-4">
          <div className="flex items-center justify-between">
            <div>
              <p className="text-sm text-gray-600">Total Crimes</p>
              <p className="text-2xl font-bold">{stats.total_crimes}</p>
            </div>
            <Database className="h-8 w-8 text-blue-600" />
          </div>
        </CardContent>
      </Card>
      
      <Card>
        <CardContent className="p-4">
          <div className="flex items-center justify-between">
            <div>
              <p className="text-sm text-gray-600">Most Common</p>
              <p className="text-lg font-semibold">
                {stats.crime_by_type[0]?._id.replace('_', ' ') || 'N/A'}
              </p>
            </div>
            <TrendingUp className="h-8 w-8 text-orange-600" />
          </div>
        </CardContent>
      </Card>
      
      <Card>
        <CardContent className="p-4">
          <div className="flex items-center justify-between">
            <div>
              <p className="text-sm text-gray-600">Top State</p>
              <p className="text-lg font-semibold">
                {stats.crime_by_state[0]?._id || 'N/A'}
              </p>
            </div>
            <MapPin className="h-8 w-8 text-green-600" />
          </div>
        </CardContent>
      </Card>
      
      <Card>
        <CardContent className="p-4">
          <div className="flex items-center justify-between">
            <div>
              <p className="text-sm text-gray-600">Recent Activity</p>
              <p className="text-2xl font-bold">{stats.recent_crimes?.length || 0}</p>
            </div>
            <Clock className="h-8 w-8 text-purple-600" />
          </div>
        </CardContent>
      </Card>
    </div>
  );
};

function App() {
  const [activeTab, setActiveTab] = useState("map");
  const [crimeData, setCrimeData] = useState([]);
  const [userLocation, setUserLocation] = useState(null);
  const [safetyAnalysis, setSafetyAnalysis] = useState(null);
  const [crimePrediction, setCrimePrediction] = useState(null);
  const [dashboardStats, setDashboardStats] = useState(null);
  const [locations, setLocations] = useState({});
  const [crimeTypes, setCrimeTypes] = useState({});
  const [selectedState, setSelectedState] = useState("");
  const [selectedCity, setSelectedCity] = useState("");
  const [loading, setLoading] = useState(false);
  const [newCrime, setNewCrime] = useState({
    crime_type: '',
    severity: 5,
    state: '',
    city: '',
    description: ''
  });

  // Load initial data
  useEffect(() => {
    loadInitialData();
  }, []);

  const loadInitialData = async () => {
    try {
      setLoading(true);
      
      // Load locations and crime types
      const [locationsRes, crimeTypesRes, crimeDataRes, statsRes] = await Promise.all([
        axios.get(`${API}/locations`),
        axios.get(`${API}/crime-types`),
        axios.get(`${API}/crime-data?limit=200`),
        axios.get(`${API}/dashboard-stats`)
      ]);
      
      setLocations(locationsRes.data.locations);
      setCrimeTypes(crimeTypesRes.data.crime_types);
      setCrimeData(crimeDataRes.data);
      setDashboardStats(statsRes.data);
      
    } catch (error) {
      console.error('Error loading initial data:', error);
    } finally {
      setLoading(false);
    }
  };

  const getCurrentLocation = async () => {
    if (!navigator.geolocation) {
      alert('Geolocation is not supported by this browser. Please click on the map to select a location.');
      return;
    }

    setLoading(true);
    
    try {
      // Show loading state immediately
      console.log('Getting user location...');
      
      const position = await new Promise((resolve, reject) => {
        navigator.geolocation.getCurrentPosition(
          resolve,
          reject,
          {
            enableHighAccuracy: true,
            timeout: 10000,
            maximumAge: 300000 // 5 minutes
          }
        );
      });

      const location = {
        lat: position.coords.latitude,
        lng: position.coords.longitude
      };
      
      console.log('Location obtained:', location);
      setUserLocation(location);
      
      // Analyze location safety
      await analyzeLocationSafety(location);
      
      // Automatically switch to Safety Analysis tab to show results
      setActiveTab('safety');
      
      setLoading(false);
      
      // Show success message
      console.log('Location analysis completed successfully');
      
    } catch (error) {
      console.error('Error getting location:', error);
      setLoading(false);
      
      let errorMessage = 'Unable to retrieve your location. ';
      
      switch(error.code) {
        case error.PERMISSION_DENIED:
          errorMessage += 'Please allow location access in your browser settings and try again.';
          break;
        case error.POSITION_UNAVAILABLE:
          errorMessage += 'Location information is unavailable. Please try again.';
          break;
        case error.TIMEOUT:
          errorMessage += 'Location request timed out. Please try again.';
          break;
        default:
          errorMessage += 'Please click on the map to select a location instead.';
          break;
      }
      
      alert(errorMessage);
    }
  };

  const analyzeLocationSafety = async (location) => {
    try {
      setLoading(true);
      console.log('Analyzing location safety for:', location);
      
      const [safetyRes, predictionRes] = await Promise.all([
        axios.post(`${API}/safety-analysis`, {
          latitude: location.lat,
          longitude: location.lng,
          radius: 5.0
        }),
        axios.post(`${API}/predict-crime`, {
          latitude: location.lat,
          longitude: location.lng
        })
      ]);
      
      console.log('Safety analysis completed:', safetyRes.data);
      console.log('Crime prediction completed:', predictionRes.data);
      
      setSafetyAnalysis(safetyRes.data);
      setCrimePrediction(predictionRes.data);
      
      setLoading(false);
      
    } catch (error) {
      console.error('Error analyzing location safety:', error);
      setLoading(false);
      
      // Show user-friendly error message
      alert('Unable to analyze location safety at this time. Please try again or select a different location.');
      
      // Clear any existing analysis
      setSafetyAnalysis(null);
      setCrimePrediction(null);
    }
  };

  const handleMapClick = (latlng) => {
    const location = { lat: latlng.lat, lng: latlng.lng };
    setUserLocation(location);
    analyzeLocationSafety(location);
  };

  const handleSubmitCrime = async (e) => {
    e.preventDefault();
    if (!userLocation) {
      alert('Please select a location on the map first');
      return;
    }

    try {
      setLoading(true);
      await axios.post(`${API}/crime-data`, {
        ...newCrime,
        location: userLocation
      });
      
      // Refresh crime data
      const crimeDataRes = await axios.get(`${API}/crime-data?limit=200`);
      setCrimeData(crimeDataRes.data);
      
      // Reset form
      setNewCrime({
        crime_type: '',
        severity: 5,
        state: '',
        city: '',
        description: ''
      });
      
      alert('Crime data submitted successfully!');
      
    } catch (error) {
      console.error('Error submitting crime data:', error);
      alert('Error submitting crime data');
    } finally {
      setLoading(false);
    }
  };

  return (
    <div className="min-h-screen bg-gradient-to-br from-slate-50 via-blue-50 to-indigo-50">
      {/* Header */}
      <header className="bg-white shadow-lg border-b">
        <div className="max-w-7xl mx-auto px-4 py-6">
          <div className="flex items-center justify-between">
            <div className="flex items-center space-x-3">
              <Shield className="h-8 w-8 text-blue-600" />
              <div>
                <h1 className="text-3xl font-bold text-gray-900">Safe Guard</h1>
                <p className="text-sm text-gray-600">AI-Powered Crime Pattern Prediction System</p>
              </div>
            </div>
            <Button 
              onClick={getCurrentLocation} 
              disabled={loading} 
              className="flex items-center gap-2 bg-blue-600 hover:bg-blue-700 text-white px-4 py-2 rounded-lg transition-colors"
            >
              <MapPin className="h-4 w-4" />
              {loading ? 'Getting Location...' : 'Get My Location'}
            </Button>
          </div>
        </div>
      </header>

      {/* Main Content */}
      <main className="max-w-7xl mx-auto px-4 py-8">
        <Tabs value={activeTab} onValueChange={setActiveTab} className="space-y-6">
          <TabsList className="grid w-full grid-cols-4">
            <TabsTrigger value="map" className="flex items-center gap-2">
              <Map className="h-4 w-4" />
              Interactive Map
            </TabsTrigger>
            <TabsTrigger value="safety" className="flex items-center gap-2">
              <Shield className="h-4 w-4" />
              Safety Analysis
            </TabsTrigger>
            <TabsTrigger value="dashboard" className="flex items-center gap-2">
              <BarChart3 className="h-4 w-4" />
              Dashboard
            </TabsTrigger>
            <TabsTrigger value="report" className="flex items-center gap-2">
              <Users className="h-4 w-4" />
              Report Crime
            </TabsTrigger>
          </TabsList>

          <TabsContent value="map" className="space-y-6">
            <Card>
              <CardHeader>
                <CardTitle>Crime Hotspot Map</CardTitle>
                <CardDescription>
                  Click anywhere on the map to analyze location safety. Use your current location for personalized recommendations.
                </CardDescription>
              </CardHeader>
              <CardContent>
                <InteractiveMap 
                  onLocationSelect={handleMapClick}
                  crimeData={crimeData}
                  userLocation={userLocation}
                  safetyAnalysis={safetyAnalysis}
                />
                {userLocation && (
                  <div className="mt-4 p-4 bg-green-50 border border-green-200 rounded-lg">
                    <div className="flex items-center gap-2 text-green-800">
                      <MapPin className="h-4 w-4" />
                      <span className="text-sm font-medium">
                        📍 Current Location: {userLocation.lat.toFixed(4)}°, {userLocation.lng.toFixed(4)}°
                      </span>
                    </div>
                    <p className="text-xs text-green-600 mt-1">
                      Safety analysis completed. Switch to Safety Analysis tab to view detailed recommendations.
                    </p>
                    <Button 
                      size="sm" 
                      className="mt-2 text-xs bg-green-600 hover:bg-green-700"
                      onClick={() => setActiveTab('safety')}
                    >
                      View Safety Analysis
                    </Button>
                  </div>
                )}
              </CardContent>
            </Card>
          </TabsContent>

          <TabsContent value="safety" className="space-y-6">
            {loading ? (
              <Card>
                <CardContent className="py-12 text-center">
                  <div className="animate-spin rounded-full h-12 w-12 border-b-2 border-blue-600 mx-auto mb-4"></div>
                  <h3 className="text-lg font-semibold mb-2">Analyzing Location Safety</h3>
                  <p className="text-gray-600">
                    Please wait while we analyze crime patterns and generate safety recommendations for your location...
                  </p>
                </CardContent>
              </Card>
            ) : userLocation && (safetyAnalysis || crimePrediction) ? (
              <div className="space-y-4">
                <Card className="bg-blue-50 border-blue-200">
                  <CardContent className="py-3">
                    <div className="flex items-center gap-2 text-blue-800">
                      <MapPin className="h-4 w-4" />
                      <span className="text-sm font-medium">
                        Analysis for: {userLocation.lat.toFixed(4)}°, {userLocation.lng.toFixed(4)}°
                      </span>
                    </div>
                  </CardContent>
                </Card>
                <SafetyRecommendations 
                  analysis={safetyAnalysis} 
                  prediction={crimePrediction} 
                />
              </div>
            ) : (
              <Card>
                <CardContent className="py-12 text-center">
                  <MapPin className="h-12 w-12 text-gray-400 mx-auto mb-4" />
                  <h3 className="text-lg font-semibold mb-2">Get Location-Based Safety Analysis</h3>
                  <p className="text-gray-600 mb-4">
                    Choose a location to get comprehensive safety analysis and AI-powered crime predictions.
                  </p>
                  <div className="space-y-3">
                    <Button 
                      onClick={getCurrentLocation} 
                      disabled={loading}
                      className="bg-blue-600 hover:bg-blue-700 text-white"
                    >
                      <MapPin className="h-4 w-4 mr-2" />
                      Use My Current Location
                    </Button>
                    <p className="text-sm text-gray-500">or</p>
                    <Button 
                      variant="outline" 
                      onClick={() => setActiveTab("map")}
                    >
                      Click on Map to Select Location
                    </Button>
                  </div>
                </CardContent>
              </Card>
            )}
          </TabsContent>

          <TabsContent value="dashboard" className="space-y-6">
            <DashboardStats stats={dashboardStats} />
            
            <div className="grid grid-cols-1 lg:grid-cols-2 gap-6">
              <Card>
                <CardHeader>
                  <CardTitle>Crime Distribution by Type</CardTitle>
                </CardHeader>
                <CardContent>
                  <div className="space-y-3">
                    {dashboardStats?.crime_by_type.slice(0, 6).map((item, index) => (
                      <div key={index} className="flex justify-between items-center">
                        <span className="text-sm capitalize">{item._id.replace('_', ' ')}</span>
                        <div className="flex items-center gap-2">
                          <div className="w-20 bg-gray-200 rounded-full h-2">
                            <div 
                              className="bg-blue-600 h-2 rounded-full" 
                              style={{ width: `${(item.count / dashboardStats?.total_crimes) * 100}%` }}
                            ></div>
                          </div>
                          <span className="text-sm font-medium">{item.count}</span>
                        </div>
                      </div>
                    ))}
                  </div>
                </CardContent>
              </Card>

              <Card>
                <CardHeader>
                  <CardTitle>Crime Distribution by State</CardTitle>
                </CardHeader>
                <CardContent>
                  <div className="space-y-3">
                    {dashboardStats?.crime_by_state.slice(0, 6).map((item, index) => (
                      <div key={index} className="flex justify-between items-center">
                        <span className="text-sm">{item._id}</span>
                        <div className="flex items-center gap-2">
                          <div className="w-20 bg-gray-200 rounded-full h-2">
                            <div 
                              className="bg-green-600 h-2 rounded-full" 
                              style={{ width: `${(item.count / dashboardStats?.total_crimes) * 100}%` }}
                            ></div>
                          </div>
                          <span className="text-sm font-medium">{item.count}</span>
                        </div>
                      </div>
                    ))}
                  </div>
                </CardContent>
              </Card>
            </div>
          </TabsContent>

          <TabsContent value="report" className="space-y-6">
            <Card>
              <CardHeader>
                <CardTitle>Report Crime Incident</CardTitle>
                <CardDescription>
                  Help improve community safety by reporting crime incidents in your area.
                </CardDescription>
              </CardHeader>
              <CardContent>
                <form onSubmit={handleSubmitCrime} className="space-y-4">
                  <div className="grid grid-cols-1 md:grid-cols-2 gap-4">
                    <div className="space-y-2">
                      <label className="text-sm font-medium">Crime Type</label>
                      <Select 
                        value={newCrime.crime_type} 
                        onValueChange={(value) => setNewCrime({...newCrime, crime_type: value})}
                      >
                        <SelectTrigger>
                          <SelectValue placeholder="Select crime type" />
                        </SelectTrigger>
                        <SelectContent>
                          {Object.keys(crimeTypes).map((type) => (
                            <SelectItem key={type} value={type}>
                              {type.replace('_', ' ').toUpperCase()}
                            </SelectItem>
                          ))}
                        </SelectContent>
                      </Select>
                    </div>
                    
                    <div className="space-y-2">
                      <label className="text-sm font-medium">Severity (1-10)</label>
                      <Input 
                        type="number" 
                        min="1" 
                        max="10" 
                        value={newCrime.severity}
                        onChange={(e) => setNewCrime({...newCrime, severity: parseInt(e.target.value)})}
                      />
                    </div>
                  </div>
                  
                  <div className="grid grid-cols-1 md:grid-cols-2 gap-4">
                    <div className="space-y-2">
                      <label className="text-sm font-medium">State</label>
                      <Select 
                        value={newCrime.state} 
                        onValueChange={(value) => setNewCrime({...newCrime, state: value, city: ''})}
                      >
                        <SelectTrigger>
                          <SelectValue placeholder="Select state" />
                        </SelectTrigger>
                        <SelectContent>
                          {Object.keys(locations).map((state) => (
                            <SelectItem key={state} value={state}>
                              {state}
                            </SelectItem>
                          ))}
                        </SelectContent>
                      </Select>
                    </div>
                    
                    <div className="space-y-2">
                      <label className="text-sm font-medium">City</label>
                      <Select 
                        value={newCrime.city} 
                        onValueChange={(value) => setNewCrime({...newCrime, city: value})}
                        disabled={!newCrime.state}
                      >
                        <SelectTrigger>
                          <SelectValue placeholder="Select city" />
                        </SelectTrigger>
                        <SelectContent>
                          {newCrime.state && locations[newCrime.state]?.map((city) => (
                            <SelectItem key={city} value={city}>
                              {city}
                            </SelectItem>
                          ))}
                        </SelectContent>
                      </Select>
                    </div>
                  </div>
                  
                  <div className="space-y-2">
                    <label className="text-sm font-medium">Description</label>
                    <Textarea 
                      placeholder="Provide additional details about the incident..."
                      value={newCrime.description}
                      onChange={(e) => setNewCrime({...newCrime, description: e.target.value})}
                    />
                  </div>
                  
                  {!userLocation && (
                    <Alert>
                      <AlertTriangle className="h-4 w-4" />
                      <AlertDescription>
                        Please select a location on the map first before submitting the report.
                      </AlertDescription>
                    </Alert>
                  )}
                  
                  <Button type="submit" disabled={loading || !userLocation} className="w-full">
                    {loading ? 'Submitting...' : 'Submit Report'}
                  </Button>
                </form>
              </CardContent>
            </Card>
          </TabsContent>
        </Tabs>
      </main>
    </div>
  );
}

export default App;