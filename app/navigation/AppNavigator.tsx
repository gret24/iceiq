/**
 * IceIQ App Navigation
 * 
 * 화면 네비게이션 & 라우팅
 */

import React from 'react';
import { NavigationContainer } from '@react-navigation/native';
import { createNativeStackNavigator } from '@react-navigation/native-stack';
import { createBottomTabNavigator } from '@react-navigation/bottom-tabs';
import { Ionicons } from '@expo/vector-icons';

// 화면 임포트
import HomeScreen from '../screens/HomeScreen';
import GameAnalysisScreen from '../screens/GameAnalysisScreen';
import HighlightShareScreen from '../screens/HighlightShareScreen';
import RosterScreen from '../screens/RosterScreen';
import PlayerProfileScreen from '../screens/PlayerProfileScreen';
import SettingsScreen from '../screens/SettingsScreen';

const Stack = createNativeStackNavigator();
const Tab = createBottomTabNavigator();

/**
 * 분석 탭 스택
 */
const AnalysisStackNavigator = () => {
  return (
    <Stack.Navigator
      screenOptions={{
        headerStyle: {
          backgroundColor: '#f5f5f5',
        },
        headerTitleStyle: {
          fontWeight: '700',
        },
        headerShadowVisible: false,
      }}
    >
      <Stack.Screen
        name="GameAnalysis"
        component={GameAnalysisScreen}
        options={{ title: '경기 분석' }}
      />
      <Stack.Screen
        name="HighlightShare"
        component={HighlightShareScreen}
        options={{
          title: '하이라이트 공유',
          headerBackTitle: '돌아가기',
        }}
      />
    </Stack.Navigator>
  );
};

/**
 * 로스터 탭 스택
 */
const RosterStackNavigator = () => {
  return (
    <Stack.Navigator
      screenOptions={{
        headerStyle: {
          backgroundColor: '#f5f5f5',
        },
        headerTitleStyle: {
          fontWeight: '700',
        },
        headerShadowVisible: false,
      }}
    >
      <Stack.Screen
        name="RosterList"
        component={RosterScreen}
        options={{ title: '로스터' }}
      />
      <Stack.Screen
        name="PlayerProfile"
        component={PlayerProfileScreen}
        options={{
          title: '선수 프로필',
          headerBackTitle: '돌아가기',
        }}
      />
    </Stack.Navigator>
  );
};

/**
 * 메인 탭 네비게이터
 */
const TabNavigator = () => {
  return (
    <Tab.Navigator
      screenOptions={({ route }) => ({
        headerShown: false,
        tabBarIcon: ({ focused, color, size }) => {
          let iconName: keyof typeof Ionicons.glyphMap = 'home';

          if (route.name === 'Home') {
            iconName = focused ? 'home' : 'home-outline';
          } else if (route.name === 'AnalysisStack') {
            iconName = focused ? 'stats-chart' : 'stats-chart-outline';
          } else if (route.name === 'RosterStack') {
            iconName = focused ? 'people' : 'people-outline';
          } else if (route.name === 'Settings') {
            iconName = focused ? 'settings' : 'settings-outline';
          }

          return <Ionicons name={iconName} size={size} color={color} />;
        },
        tabBarActiveTintColor: '#1f77d2',
        tabBarInactiveTintColor: '#999',
        tabBarStyle: {
          borderTopWidth: 1,
          borderTopColor: '#eee',
          paddingBottom: 5,
          height: 60,
        },
        tabBarLabelStyle: {
          fontSize: 12,
          marginTop: 4,
        },
      })}
    >
      <Tab.Screen
        name="Home"
        component={HomeScreen}
        options={{ tabBarLabel: '홈' }}
      />
      <Tab.Screen
        name="AnalysisStack"
        component={AnalysisStackNavigator}
        options={{ tabBarLabel: '분석' }}
      />
      <Tab.Screen
        name="RosterStack"
        component={RosterStackNavigator}
        options={{ tabBarLabel: '로스터' }}
      />
      <Tab.Screen
        name="Settings"
        component={SettingsScreen}
        options={{ tabBarLabel: '설정' }}
      />
    </Tab.Navigator>
  );
};

/**
 * 루트 네비게이터
 */
export const RootNavigator = () => {
  return (
    <NavigationContainer>
      <Stack.Navigator
        screenOptions={{
          headerShown: false,
          cardStyle: { backgroundColor: '#fff' },
        }}
      >
        <Stack.Screen name="MainTab" component={TabNavigator} />
      </Stack.Navigator>
    </NavigationContainer>
  );
};

export default RootNavigator;
