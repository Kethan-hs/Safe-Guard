// src/App.js
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
  XCircle,
  Activity,
  Eye,
  Zap
} from "lucide-react";
import L from 'leaflet';

// Fix for default markers in react-leaflet
delete L.Icon.Default.prototype._getIconUrl;
L.Icon.Default.mergeOptions({
  iconRetinaUrl: 'https://cdnjs.cloudflare.com/ajax/libs/leaflet/1.7.1/images/marker-icon-2x.png',
  iconUrl: 'https://cdnjs.cloudflare.com/ajax/libs/leaflet/1.7.1/images/marker-icon.png',
  shadowUrl: 'https://cdnjs.cloudflare.com/ajax/libs/leaflet/1.7.1/images/marker-shadow.png',
});

const BACKEND_URL = process.env.REACT_APP_BACKEND_URL || "http://localhost:8000";
const API = `${BACKEND_URL}/api`;

// ========== Enhanced Loading Component ==========
const EnhancedLoading = ({ message = "Analyzing..." }) => (
  <div className="glass-card p-8 text-center">
    <div className="relative mx-auto w-16 h-16 mb-6">
      <div className="absolute inset-0 rounded-full border-4 border-blue-200 animate-pulse"></div>
      <div className="absolute inset-0 rounded-full border-t-4 border-blue-500 animate-spin"></div>
      <div className="absolute inset-2 rounded-full border-2 border-purple-300 animate-ping opacity-20"></div>
    </div>
    <div className="space-y-2">
      <h3 className="text-lg font-semibold bg-gradient-to-r from-blue-600 to-purple-600 bg-clip-text text-transparent">
        {message}
      </h3>
      <p className="text-gray-600 text-sm">
        Processing crime patterns with AI intelligence...
      </p>
    </div>
  </div>
);

// ========== Interactive Map Component ==========
const InteractiveMap = ({ onLocationSelect, crimeData = [], userLocation, safetyAnalysis }) => {
  const LocationMarker = () => {
    useMapEvents({
      click(e) {
        onLocationSelect(e.latlng);
      },
    });
    return null;
  };

  const getRiskColor = (severity = 0) => {
    if (severity >= 8) return '#dc2626';
    if (severity >= 6) return '#ea580c';
    if (severity >= 4) return '#ca8a04';
    return '#16a34a';
  };

  const getRiskColorFromLevel = (level) => {
    switch (level) {
      case 'critical': return '#dc2626';
      case 'high': return '#ea580c';
      case 'medium': return '#ca8a04';
      case 'low': return '#16a34a';
      default: return '#6b7280';
    }
  };

  return (
    <div className="relative overflow-hidden rounded-2xl">
      <div className="absolute inset-0 bg-gradient-to-br from-blue-500/20 to-purple-500/20 z-10 pointer-events-none"></div>
      <MapContainer
        center={[20.5937, 78.9629]}
        zoom={5}
        style={{ height: '500px', width: '100%' }}
        className="rounded-2xl map-container"
      >
        <TileLayer
          url="https://{s}.tile.openstreetmap.org/{z}/{x}/{y}.png"
          attribution='&copy; <a href="https://www.openstreetmap.org/copyright">OpenStreetMap</a> contributors'
        />
        <LocationMarker />

        {Array.isArray(crimeData) && crimeData.map((crime, index) => {
          const lat = crime?.location?.lat ?? 0;
          const lng = crime?.location?.lng ?? 0;
          const severity = crime?.severity ?? 0;
          const crimeType = (crime?.crime_type || "").replace('_', ' ').toUpperCase();

          return (
            <Marker
              key={index}
              position={[lat, lng]}
              icon={L.divIcon({
                className: 'custom-crime-marker',
                html: `<div class="crime-marker" style="background: linear-gradient(45deg, ${getRiskColor(severity)}, ${getRiskColor(severity)}aa); box-shadow: 0 4px 12px rgba(0,0,0,0.3), 0 0 0 2px rgba(255,255,255,0.8);"></div>`,
                iconSize: [14, 14],
                iconAnchor: [7, 7],
              })}
            >
              <Popup>
                <div className="p-3 space-y-2">
                  <h4 className="font-bold text-sm text-gray-800">{crimeType}</h4>
                  <p className="text-xs text-gray-600">{crime?.city ?? 'Unknown'}, {crime?.state ?? 'Unknown'}</p>
                  <div className="flex items-center gap-2">
                    <div className="flex-1 bg-gray-200 rounded-full h-1.5">
                      <div 
                        className="h-1.5 rounded-full bg-gradient-to-r from-yellow-400 to-red-500"
                        style={{ width: `${(severity / 10) * 100}%` }}
                      ></div>
                    </div>
                    <span className="text-xs font-medium">{severity}/10</span>
                  </div>
                  <p className="text-xs text-gray-700">{crime?.description ?? ''}</p>
                </div>
              </Popup>
            </Marker>
          );
        })}

        {userLocation && (
          <>
            <Marker
              position={[userLocation.lat, userLocation.lng]}
              icon={L.divIcon({
                className: 'user-location-marker',
                html: '<div class="user-marker"></div>',
                iconSize: [20, 20],
                iconAnchor: [10, 10]
              })}
            >
              <Popup>
                <div className="p-2 text-center">
                  <div className="flex items-center gap-2 text-blue-600">
                    <MapPin className="h-4 w-4" />
                    <span className="font-medium">Your Location</span>
                  </div>
                </div>
              </Popup>
            </Marker>

            {safetyAnalysis && (
              <Circle
                center={[userLocation.lat, userLocation.lng]}
                radius={5000}
                pathOptions={{
                  color: getRiskColorFromLevel(safetyAnalysis?.risk_level),
                  weight: 3,
                  opacity: 0.8,
                  fillOpacity: 0.1,
                  dashArray: '10,5'
                }}
              />
            )}
          </>
        )}
      </MapContainer>
    </div>
  );
};

// ========== Enhanced Safety Recommendations Component ==========
const SafetyRecommendations = ({ analysis, prediction }) => {
  if (!analysis) return null;

  const getRiskBadgeVariant = (level) => {
    switch (level) {
      case 'critical': return 'destructive';
      case 'high': return 'destructive';
      case 'medium': return 'default';
      case 'low': return 'secondary';
      default: return 'outline';
    }
  };

  const getRiskIcon = (level) => {
    switch (level) {
      case 'critical': return <XCircle className="h-5 w-5" />;
      case 'high': return <AlertTriangle className="h-5 w-5" />;
      case 'medium': return <AlertCircle className="h-5 w-5" />;
      case 'low': return <CheckCircle2 className="h-5 w-5" />;
      default: return <Shield className="h-5 w-5" />;
    }
  };

  const getRiskGradient = (level) => {
    switch (level) {
      case 'critical': return 'from-red-500 to-red-600';
      case 'high': return 'from-orange-500 to-red-500';
      case 'medium': return 'from-yellow-500 to-orange-500';
      case 'low': return 'from-green-500 to-blue-500';
      default: return 'from-gray-500 to-gray-600';
    }
  };

  return (
    <div className="space-y-6">
      <div className="glass-card hover-lift-subtle">
        <CardHeader className="pb-4">
          <CardTitle className="flex items-center gap-3">
            <div className="p-2 rounded-xl bg-gradient-to-br from-blue-500/20 to-purple-500/20">
              <Shield className="h-5 w-5 text-blue-600" />
            </div>
            <div>
              <h3 className="bg-gradient-to-r from-blue-600 to-purple-600 bg-clip-text text-transparent">
                Location Safety Analysis
              </h3>
              <p className="text-sm text-gray-600 font-normal mt-1">
                AI-powered recommendations based on local patterns
              </p>
            </div>
          </CardTitle>
        </CardHeader>
        <CardContent className="space-y-6">
          <div className="relative p-6 rounded-2xl bg-gradient-to-br from-white to-gray-50 border border-white/50 shadow-lg">
            <div className="flex items-center justify-between mb-4">
              <div className="flex items-center gap-3">
                <div className={`p-2 rounded-xl bg-gradient-to-br ${getRiskGradient(analysis?.risk_level)} shadow-lg`}>
                  <div className="text-white">
                    {getRiskIcon(analysis?.risk_level)}
                  </div>
                </div>
                <div>
                  <span className="text-lg font-bold text-gray-900">Risk Assessment</span>
                  <div className="flex items-center gap-2 mt-1">
                    <Badge variant={getRiskBadgeVariant(analysis?.risk_level)} className="text-xs font-semibold">
                      {analysis?.risk_level?.toUpperCase() ?? 'N/A'}
                    </Badge>
                  </div>
                </div>
              </div>
              <div className="text-right">
                <div className="text-3xl font-bold bg-gradient-to-r from-blue-600 to-purple-600 bg-clip-text text-transparent">
                  {Math.round(analysis?.risk_score ?? 0)}
                </div>
                <div className="text-xs text-gray-500 font-medium">RISK SCORE</div>
              </div>
            </div>

            <div className="space-y-3">
              <div className="flex justify-between text-sm text-gray-700">
                <span>Threat Level</span>
                <span className="font-semibold">{Math.round(analysis?.risk_score ?? 0)}/100</span>
              </div>
              <div className="relative h-3 bg-gray-200 rounded-full overflow-hidden">
                <div 
                  className={`h-full bg-gradient-to-r ${getRiskGradient(analysis?.risk_level)} shadow-inner transition-all duration-1000 ease-out`}
                  style={{ width: `${analysis?.risk_score ?? 0}%` }}
                />
                <div className="absolute inset-0 bg-gradient-to-r from-transparent via-white/20 to-transparent animate-shimmer" />
              </div>
            </div>
          </div>

          <Separator className="bg-gradient-to-r from-transparent via-gray-300 to-transparent" />

          <div className="space-y-4">
            <h4 className="font-bold text-gray-900 flex items-center gap-2">
              <Eye className="h-4 w-4 text-blue-500" />
              Safety Recommendations
            </h4>
            <div className="grid gap-3">
              {Array.isArray(analysis?.recommendations) && analysis.recommendations.map((rec, index) => (
                <div key={index} className="p-4 rounded-xl bg-gradient-to-r from-blue-50 to-purple-50 border border-blue-200/50 hover-lift-micro">
                  <div className="flex items-start gap-3">
                    <div className="p-1 rounded-lg bg-blue-500/20">
                      <Zap className="h-3 w-3 text-blue-600" />
                    </div>
                    <p className="text-sm text-gray-700 flex-1">{rec}</p>
                  </div>
                </div>
              ))}
            </div>
          </div>

          {Array.isArray(analysis?.nearby_crimes) && analysis.nearby_crimes.length > 0 && (
            <>
              <Separator className="bg-gradient-to-r from-transparent via-gray-300 to-transparent" />
              <div className="space-y-4">
                <h4 className="font-bold text-gray-900 flex items-center gap-2">
                  <Activity className="h-4 w-4 text-orange-500" />
                  Recent Nearby Incidents ({analysis.nearby_crimes.length})
                </h4>
                <div className="grid gap-2 max-h-48 overflow-y-auto custom-scrollbar">
                  {analysis.nearby_crimes.slice(0, 5).map((crime, index) => (
                    <div key={index} className="p-3 rounded-xl bg-white border border-gray-200/50 shadow-sm hover-lift-micro">
                      <div className="flex justify-between items-start mb-2">
                        <span className="font-semibold text-sm text-gray-800">
                          {(crime.crime_type || '').replace('_', ' ')}
                        </span>
                        <Badge variant="outline" className="text-xs bg-orange-50 text-orange-600 border-orange-200">
                          {crime.distance ?? 0}km away
                        </Badge>
                      </div>
                      <div className="flex items-center gap-2">
                        <div className="flex-1 bg-gray-200 rounded-full h-1.5">
                          <div 
                            className="h-1.5 rounded-full bg-gradient-to-r from-yellow-400 to-red-500"
                            style={{ width: `${((crime.severity ?? 0) / 10) * 100}%` }}
                          />
                        </div>
                        <span className="text-xs text-gray-600 font-medium">
                          {crime.severity ?? 0}/10
                        </span>
                      </div>
                    </div>
                  ))}
                </div>
              </div>
            </>
          )}
        </CardContent>
      </div>

      {prediction && (
        <div className="glass-card hover-lift-subtle">
          <CardHeader className="pb-4">
            <CardTitle className="flex items-center gap-3">
              <div className="p-2 rounded-xl bg-gradient-to-br from-purple-500/20 to-pink-500/20">
                <Target className="h-5 w-5 text-purple-600" />
              </div>
              <div>
                <h3 className="bg-gradient-to-r from-purple-600 to-pink-600 bg-clip-text text-transparent">
                  AI Crime Prediction
                </h3>
                <p className="text-sm text-gray-600 font-normal mt-1">
                  Machine learning risk assessment
                </p>
              </div>
            </CardTitle>
          </CardHeader>
          <CardContent className="space-y-4">
            <div className="p-4 rounded-xl bg-gradient-to-br from-purple-50 to-pink-50 border border-purple-200/50">
              <div className="space-y-3">
                <div className="flex justify-between text-sm">
                  <span className="text-gray-700">Predicted Risk Score</span>
                  <span className="font-bold text-purple-600">
                    {Math.round(prediction?.predicted_risk_score ?? 0)}/100
                  </span>
                </div>
                <div className="relative h-3 bg-purple-200 rounded-full overflow-hidden">
                  <div 
                    className="h-full bg-gradient-to-r from-purple-500 to-pink-500 transition-all duration-1000 ease-out"
                    style={{ width: `${prediction?.predicted_risk_score ?? 0}%` }}
                  />
                </div>
              </div>
            </div>

            <div className="space-y-3">
              <span className="text-sm font-bold text-gray-900">Likely Crime Types</span>
              <div className="flex flex-wrap gap-2">
                {Array.isArray(prediction?.predicted_crime_types) && prediction.predicted_crime_types.map((type, index) => (
                  <Badge 
                    key={index} 
                    variant="outline" 
                    className="text-xs bg-gradient-to-r from-purple-50 to-pink-50 border-purple-200 text-purple-700 hover-lift-micro"
                  >
                    {(type || '').replace('_', ' ')}
                  </Badge>
                ))}
              </div>
            </div>

            <div className="flex items-center justify-between p-3 rounded-xl bg-gradient-to-r from-green-50 to-blue-50 border border-green-200/50">
              <span className="text-sm font-medium text-gray-700">AI Confidence</span>
              <span className="text-sm font-bold text-green-600">
                {Math.round((prediction?.confidence ?? 0) * 100)}%
              </span>
            </div>
          </CardContent>
        </div>
      )}
    </div>
  );
};

// ========== Enhanced Dashboard Stats Component ==========
const DashboardStats = ({ stats }) => {
  if (!stats) return null;

  const mostCommon = (stats.crime_by_type && stats.crime_by_type[0]?._id?.replace('_', ' ')) || 'N/A';
  const topState = (stats.crime_by_state && stats.crime_by_state[0]?._id) || 'N/A';
  const recentCount = stats.recent_crimes ? stats.recent_crimes.length : 0;

  const statCards = [
    {
      title: "Total Crimes",
      value: stats.total_crimes ?? 0,
      icon: Database,
      gradient: "from-blue-500 to-cyan-500",
      bg: "from-blue-50 to-cyan-50"
    },
    {
      title: "Most Common",
      value: mostCommon,
      icon: TrendingUp,
      gradient: "from-orange-500 to-red-500",
      bg: "from-orange-50 to-red-50"
    },
    {
      title: "Top State",
      value: topState,
      icon: MapPin,
      gradient: "from-green-500 to-emerald-500",
      bg: "from-green-50 to-emerald-50"
    },
    {
      title: "Recent Activity",
      value: recentCount,
      icon: Clock,
      gradient: "from-purple-500 to-pink-500",
      bg: "from-purple-50 to-pink-50"
    }
  ];

  return (
    <div className="grid grid-cols-1 md:grid-cols-2 lg:grid-cols-4 gap-6 mb-8">
      {statCards.map((card, index) => (
        <div key={index} className="glass-card hover-lift group">
          <CardContent className="p-6">
            <div className="flex items-center justify-between">
              <div className="space-y-2">
                <p className="text-sm font-medium text-gray-600">{card.title}</p>
                <p className={`text-2xl font-bold bg-gradient-to-r ${card.gradient} bg-clip-text text-transparent`}>
                  {typeof card.value === 'number' && card.value > 999 
                    ? `${(card.value / 1000).toFixed(1)}k` 
                    : card.value}
                </p>
              </div>
              <div className={`p-3 rounded-2xl bg-gradient-to-br ${card.bg} group-hover:scale-110 transition-transform duration-300`}>
                <card.icon className={`h-6 w-6 bg-gradient-to-r ${card.gradient} bg-clip-text text-transparent`} />
              </div>
            </div>
          </CardContent>
        </div>
      ))}
    </div>
  );
};

// ========== Main App ==========
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

  useEffect(() => {
    loadInitialData();
  }, []);

  const loadInitialData = async () => {
    try {
      setLoading(true);

      const [locationsRes, crimeTypesRes, crimeDataRes, statsRes] = await Promise.all([
        axios.get(`${API}/locations`),
        axios.get(`${API}/crime-types`),
        axios.get(`${API}/crime-data?limit=200`),
        axios.get(`${API}/dashboard-stats`)
      ]);

      setLocations(locationsRes.data?.locations ?? {});
      setCrimeTypes(crimeTypesRes.data?.crime_types ?? {});
      setCrimeData(Array.isArray(crimeDataRes.data) ? crimeDataRes.data : (crimeDataRes.data?.crimes ?? []));
      setDashboardStats(statsRes.data ?? { total_crimes: 0, crime_by_type: [], crime_by_state: [], recent_crimes: [] });

    } catch (error) {
      console.error('Error loading initial data:', error);
      alert('Error loading initial data from backend. Check the backend server and REACT_APP_BACKEND_URL.');
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
      const position = await new Promise((resolve, reject) => {
        navigator.geolocation.getCurrentPosition(resolve, reject, {
          enableHighAccuracy: true,
          timeout: 10000,
          maximumAge: 300000,
        });
      });

      const location = {
        lat: position.coords.latitude,
        lng: position.coords.longitude
      };
      setUserLocation(location);
      await analyzeLocationSafety(location);
      setActiveTab('safety');
    } catch (err) {
      console.error('Error getting location:', err);
      let message = 'Unable to retrieve your location. ';
      if (err?.code === 1) message += 'Permission denied. Please allow location access.';
      else if (err?.code === 2) message += 'Position unavailable.';
      else if (err?.code === 3) message += 'Timeout. Please try again.';
      else message += 'You can click on the map to select a location.';
      alert(message);
    } finally {
      setLoading(false);
    }
  };

  const analyzeLocationSafety = async (locationRequest) => {
    try {
      setLoading(true);
      const safetyRes = await axios.post(`${API}/safety-analysis`, {
        latitude: locationRequest.lat,
        longitude: locationRequest.lng,
        radius: 5.0
      });
      setSafetyAnalysis(safetyRes.data);

      const predRes = await axios.post(`${API}/predict-crime`, {
        latitude: locationRequest.lat,
        longitude: locationRequest.lng
      });
      setCrimePrediction(predRes.data);

    } catch (error) {
      console.error('Error analyzing location safety:', error);
      alert('Unable to analyze location safety at this time. Please try again later.');
      setSafetyAnalysis(null);
      setCrimePrediction(null);
    } finally {
      setLoading(false);
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

      const crimeDataRes = await axios.get(`${API}/crime-data?limit=200`);
      setCrimeData(Array.isArray(crimeDataRes.data) ? crimeDataRes.data : (crimeDataRes.data?.crimes ?? []));

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
      alert('Error submitting crime data. Check console for details.');
    } finally {
      setLoading(false);
    }
  };

  const crimeTypeKeys = Array.isArray(Object.keys(crimeTypes || {})) ? Object.keys(crimeTypes || {}) : [];

  return (
    <div className="min-h-screen bg-gradient-to-br from-slate-900 via-blue-900 to-indigo-900 relative overflow-x-hidden">
      {/* Animated Background Elements */}
      <div className="fixed inset-0 overflow-hidden pointer-events-none">
        <div className="absolute top-10 left-10 w-72 h-72 bg-blue-500/10 rounded-full blur-3xl animate-pulse"></div>
        <div className="absolute top-1/2 right-10 w-96 h-96 bg-purple-500/10 rounded-full blur-3xl animate-pulse delay-700"></div>
        <div className="absolute bottom-10 left-1/3 w-80 h-80 bg-indigo-500/10 rounded-full blur-3xl animate-pulse delay-1000"></div>
      </div>

      {/* Header */}
      <header className="relative z-10 backdrop-blur-md bg-white/10 border-b border-white/20 sticky top-0">
        <div className="max-w-7xl mx-auto px-6 py-6">
          <div className="flex items-center justify-between">
            <div className="flex items-center space-x-4">
              <div className="p-3 rounded-2xl bg-gradient-to-br from-blue-500/20 to-purple-500/20 backdrop-blur-sm border border-white/20">
                <Shield className="h-8 w-8 text-white" />
              </div>
              <div>
                <h1 className="text-4xl font-bold bg-gradient-to-r from-white to-blue-200 bg-clip-text text-transparent">
                  SafeGuard
                </h1>
                <p className="text-blue-200/80 text-sm font-medium">
                  AI-Powered Crime Pattern Prediction System
                </p>
              </div>
            </div>
            <Button
              onClick={getCurrentLocation}
              disabled={loading}
              className="glass-button hover-lift-subtle group"
            >
              <MapPin className="h-4 w-4 mr-2 group-hover:scale-110 transition-transform" />
              {loading ? 'Getting Location...' : 'Get My Location'}
            </Button>
          </div>
        </div>
      </header>

      {/* Main */}
      <main className="relative z-10 max-w-7xl mx-auto px-6 py-8">
        <Tabs value={activeTab} onValueChange={setActiveTab} className="space-y-8">
          <div className="glass-card p-2">
            <TabsList className="grid w-full grid-cols-4 bg-transparent">
              {[
                { value: "map", label: "Interactive Map", icon: Map },
                { value: "safety", label: "Safety Analysis", icon: Shield },
                { value: "dashboard", label: "Dashboard", icon: BarChart3 },
                { value: "report", label: "Report Crime", icon: Users }
              ].map((tab) => (
                <TabsTrigger 
                  key={tab.value} 
                  value={tab.value} 
                  className="tab-trigger group"
                >
                  <tab.icon className="h-4 w-4 mr-2 group-hover:scale-110 transition-transform" />
                  {tab.label}
                </TabsTrigger>
              ))}
            </TabsList>
          </div>

          <TabsContent value="map" className="space-y-6">
            <div className="glass-card">
              <CardHeader className="pb-4">
                <CardTitle className="flex items-center gap-3">
                  <div className="p-2 rounded-xl bg-gradient-to-br from-green-500/20 to-blue-500/20">
                    <Map className="h-5 w-5 text-green-600" />
                  </div>
                  <div>
                    <h3 className="bg-gradient-to-r from-green-600 to-blue-600 bg-clip-text text-transparent">
                      Crime Hotspot Map
                    </h3>
                    <p className="text-sm text-gray-600 font-normal mt-1">
                      Click anywhere to analyze location safety with AI intelligence
                    </p>
                  </div>
                </CardTitle>
              </CardHeader>
              <CardContent>
                <InteractiveMap
                  onLocationSelect={handleMapClick}
                  crimeData={crimeData}
                  userLocation={userLocation}
                  safetyAnalysis={safetyAnalysis}
                />
                {userLocation && (
                  <div className="mt-6 p-6 rounded-2xl bg-gradient-to-r from-green-50 to-blue-50 border border-green-200/50 hover-lift-micro">
                    <div className="flex items-center gap-3 text-green-800 mb-3">
                      <div className="p-2 rounded-xl bg-green-500/20">
                        <MapPin className="h-4 w-4 text-green-600" />
                      </div>
                      <div>
                        <span className="text-sm font-bold">
                          Current Location: {userLocation.lat.toFixed(4)}, {userLocation.lng.toFixed(4)}
                        </span>
                        <p className="text-xs text-green-600 mt-1">
                          Safety analysis completed. View detailed recommendations below.
                        </p>
                      </div>
                    </div>
                    <Button
                      size="sm"
                      className="glass-button-sm hover-lift-micro"
                      onClick={() => setActiveTab('safety')}
                    >
                      <Shield className="h-3 w-3 mr-2" />
                      View Safety Analysis
                    </Button>
                  </div>
                )}
              </CardContent>
            </div>
          </TabsContent>

          <TabsContent value="safety" className="space-y-6">
            {loading ? (
              <EnhancedLoading message="Analyzing Location Safety" />
            ) : userLocation && (safetyAnalysis || crimePrediction) ? (
              <div className="space-y-6">
                <div className="glass-card p-4">
                  <div className="flex items-center gap-3 text-blue-800">
                    <div className="p-2 rounded-xl bg-blue-500/20">
                      <MapPin className="h-4 w-4 text-blue-600" />
                    </div>
                    <div>
                      <span className="text-sm font-bold">
                        Analysis for: {userLocation.lat.toFixed(4)}, {userLocation.lng.toFixed(4)}
                      </span>
                      <p className="text-xs text-blue-600">
                        AI-powered safety assessment with real-time data
                      </p>
                    </div>
                  </div>
                </div>
                <SafetyRecommendations analysis={safetyAnalysis} prediction={crimePrediction} />
              </div>
            ) : (
              <div className="glass-card">
                <CardContent className="py-16 text-center">
                  <div className="mb-8">
                    <div className="relative mx-auto w-20 h-20 mb-4">
                      <div className="absolute inset-0 rounded-full border-4 border-blue-200 animate-pulse"></div>
                      <div className="absolute inset-2 rounded-full bg-gradient-to-br from-blue-500 to-purple-500 flex items-center justify-center">
                        <MapPin className="h-8 w-8 text-white" />
                      </div>
                    </div>
                  </div>
                  <h3 className="text-xl font-bold bg-gradient-to-r from-blue-600 to-purple-600 bg-clip-text text-transparent mb-4">
                    Get Location-Based Safety Analysis
                  </h3>
                  <p className="text-gray-600 mb-8 max-w-md mx-auto">
                    Choose a location to get comprehensive safety analysis and AI-powered crime predictions with real-time insights.
                  </p>
                  <div className="space-y-4">
                    <Button
                      onClick={getCurrentLocation}
                      disabled={loading}
                      className="glass-button hover-lift group"
                    >
                      <MapPin className="h-4 w-4 mr-2 group-hover:scale-110 transition-transform" />
                      Use My Current Location
                    </Button>
                    <p className="text-sm text-gray-500">or</p>
                    <Button
                      variant="outline"
                      onClick={() => setActiveTab("map")}
                      className="border-white/30 bg-white/10 backdrop-blur-sm hover:bg-white/20 text-gray-700"
                    >
                      <Map className="h-4 w-4 mr-2" />
                      Click on Map to Select Location
                    </Button>
                  </div>
                </CardContent>
              </div>
            )}
          </TabsContent>

          <TabsContent value="dashboard" className="space-y-6">
            <DashboardStats stats={dashboardStats} />

            <div className="grid grid-cols-1 lg:grid-cols-2 gap-6">
              <div className="glass-card">
                <CardHeader className="pb-4">
                  <CardTitle className="flex items-center gap-3">
                    <div className="p-2 rounded-xl bg-gradient-to-br from-blue-500/20 to-indigo-500/20">
                      <BarChart3 className="h-5 w-5 text-blue-600" />
                    </div>
                    <h3 className="bg-gradient-to-r from-blue-600 to-indigo-600 bg-clip-text text-transparent">
                      Crime Distribution by Type
                    </h3>
                  </CardTitle>
                </CardHeader>
                <CardContent>
                  <div className="space-y-4">
                    {Array.isArray(dashboardStats?.crime_by_type) && dashboardStats.crime_by_type.slice(0, 6).map((item, index) => (
                      <div key={index} className="p-4 rounded-xl bg-gradient-to-r from-white to-gray-50 border border-gray-200/50 hover-lift-micro">
                        <div className="flex justify-between items-center mb-2">
                          <span className="text-sm font-semibold capitalize text-gray-800">
                            {(item._id || '').replace('_', ' ')}
                          </span>
                          <span className="text-sm font-bold text-blue-600">{item.count}</span>
                        </div>
                        <div className="flex items-center gap-3">
                          <div className="flex-1 bg-gray-200 rounded-full h-2.5 overflow-hidden">
                            <div
                              className="bg-gradient-to-r from-blue-500 to-indigo-500 h-2.5 rounded-full transition-all duration-1000 ease-out"
                              style={{ width: `${((item.count || 0) / (dashboardStats?.total_crimes || 1)) * 100}%` }}
                            />
                          </div>
                          <span className="text-xs text-gray-500 font-medium">
                            {(((item.count || 0) / (dashboardStats?.total_crimes || 1)) * 100).toFixed(1)}%
                          </span>
                        </div>
                      </div>
                    ))}
                  </div>
                </CardContent>
              </div>

              <div className="glass-card">
                <CardHeader className="pb-4">
                  <CardTitle className="flex items-center gap-3">
                    <div className="p-2 rounded-xl bg-gradient-to-br from-green-500/20 to-emerald-500/20">
                      <MapPin className="h-5 w-5 text-green-600" />
                    </div>
                    <h3 className="bg-gradient-to-r from-green-600 to-emerald-600 bg-clip-text text-transparent">
                      Crime Distribution by State
                    </h3>
                  </CardTitle>
                </CardHeader>
                <CardContent>
                  <div className="space-y-4">
                    {Array.isArray(dashboardStats?.crime_by_state) && dashboardStats.crime_by_state.slice(0, 6).map((item, index) => (
                      <div key={index} className="p-4 rounded-xl bg-gradient-to-r from-white to-gray-50 border border-gray-200/50 hover-lift-micro">
                        <div className="flex justify-between items-center mb-2">
                          <span className="text-sm font-semibold text-gray-800">{item._id}</span>
                          <span className="text-sm font-bold text-green-600">{item.count}</span>
                        </div>
                        <div className="flex items-center gap-3">
                          <div className="flex-1 bg-gray-200 rounded-full h-2.5 overflow-hidden">
                            <div
                              className="bg-gradient-to-r from-green-500 to-emerald-500 h-2.5 rounded-full transition-all duration-1000 ease-out"
                              style={{ width: `${((item.count || 0) / (dashboardStats?.total_crimes || 1)) * 100}%` }}
                            />
                          </div>
                          <span className="text-xs text-gray-500 font-medium">
                            {(((item.count || 0) / (dashboardStats?.total_crimes || 1)) * 100).toFixed(1)}%
                          </span>
                        </div>
                      </div>
                    ))}
                  </div>
                </CardContent>
              </div>
            </div>
          </TabsContent>

          <TabsContent value="report" className="space-y-6">
            <div className="glass-card">
              <CardHeader className="pb-4">
                <CardTitle className="flex items-center gap-3">
                  <div className="p-2 rounded-xl bg-gradient-to-br from-orange-500/20 to-red-500/20">
                    <Users className="h-5 w-5 text-orange-600" />
                  </div>
                  <div>
                    <h3 className="bg-gradient-to-r from-orange-600 to-red-600 bg-clip-text text-transparent">
                      Report Crime Incident
                    </h3>
                    <p className="text-sm text-gray-600 font-normal mt-1">
                      Help improve community safety by reporting incidents
                    </p>
                  </div>
                </CardTitle>
              </CardHeader>
              <CardContent>
                <form onSubmit={handleSubmitCrime} className="space-y-6">
                  <div className="grid grid-cols-1 md:grid-cols-2 gap-6">
                    <div className="space-y-3">
                      <label className="text-sm font-bold text-gray-800">Crime Type</label>
                      <div className="relative">
                        <Select
                          value={newCrime.crime_type}
                          onValueChange={(value) => setNewCrime({ ...newCrime, crime_type: value })}
                        >
                          <SelectTrigger className="glass-input">
                            <SelectValue placeholder="Select crime type" />
                          </SelectTrigger>
                          <SelectContent>
                            {crimeTypeKeys.map((type) => (
                              <SelectItem key={type} value={type}>
                                {type.replace('_', ' ').toUpperCase()}
                              </SelectItem>
                            ))}
                          </SelectContent>
                        </Select>
                      </div>
                    </div>

                    <div className="space-y-3">
                      <label className="text-sm font-bold text-gray-800">Severity (1-10)</label>
                      <div className="relative">
                        <Input
                          type="number"
                          min="1"
                          max="10"
                          value={newCrime.severity}
                          onChange={(e) => setNewCrime({ ...newCrime, severity: parseInt(e.target.value) || 1 })}
                          className="glass-input"
                        />
                        <div className="absolute right-3 top-1/2 transform -translate-y-1/2">
                          <div className="flex items-center gap-1">
                            {[...Array(Math.min(5, Math.max(1, Math.ceil(newCrime.severity / 2))))].map((_, i) => (
                              <div key={i} className="w-2 h-2 bg-gradient-to-r from-yellow-400 to-red-500 rounded-full"></div>
                            ))}
                          </div>
                        </div>
                      </div>
                    </div>
                  </div>

                  <div className="grid grid-cols-1 md:grid-cols-2 gap-6">
                    <div className="space-y-3">
                      <label className="text-sm font-bold text-gray-800">State</label>
                      <Select
                        value={newCrime.state}
                        onValueChange={(value) => setNewCrime({ ...newCrime, state: value, city: '' })}
                      >
                        <SelectTrigger className="glass-input">
                          <SelectValue placeholder="Select state" />
                        </SelectTrigger>
                        <SelectContent>
                          {Object.keys(locations || {}).map((state) => (
                            <SelectItem key={state} value={state}>
                              {state}
                            </SelectItem>
                          ))}
                        </SelectContent>
                      </Select>
                    </div>

                    <div className="space-y-3">
                      <label className="text-sm font-bold text-gray-800">City</label>
                      <Select
                        value={newCrime.city}
                        onValueChange={(value) => setNewCrime({ ...newCrime, city: value })}
                        disabled={!newCrime.state}
                      >
                        <SelectTrigger className="glass-input">
                          <SelectValue placeholder="Select city" />
                        </SelectTrigger>
                        <SelectContent>
                          {newCrime.state && (locations[newCrime.state] || []).map((city) => (
                            <SelectItem key={city} value={city}>
                              {city}
                            </SelectItem>
                          ))}
                        </SelectContent>
                      </Select>
                    </div>
                  </div>

                  <div className="space-y-3">
                    <label className="text-sm font-bold text-gray-800">Description</label>
                    <Textarea
                      placeholder="Provide additional details about the incident..."
                      value={newCrime.description}
                      onChange={(e) => setNewCrime({ ...newCrime, description: e.target.value })}
                      className="glass-input min-h-[120px]"
                    />
                  </div>

                  {!userLocation && (
                    <Alert className="border-orange-200 bg-gradient-to-r from-orange-50 to-yellow-50">
                      <AlertTriangle className="h-4 w-4 text-orange-600" />
                      <AlertDescription className="text-orange-800">
                        Please select a location on the map first before submitting the report.
                      </AlertDescription>
                    </Alert>
                  )}

                  <Button 
                    type="submit" 
                    disabled={loading || !userLocation} 
                    className="w-full glass-button hover-lift group"
                  >
                    {loading ? (
                      <>
                        <div className="animate-spin rounded-full h-4 w-4 border-b-2 border-white mr-2"></div>
                        Submitting Report...
                      </>
                    ) : (
                      <>
                        <Users className="h-4 w-4 mr-2 group-hover:scale-110 transition-transform" />
                        Submit Crime Report
                      </>
                    )}
                  </Button>
                </form>
              </CardContent>
            </div>
          </TabsContent>
        </Tabs>
      </main>
    </div>
  );
}

export default App;
                