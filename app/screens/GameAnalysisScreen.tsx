/**
 * IceIQ Game Analysis Screen
 * 
 * 경기 분석 & 하이라이트 목록
 */

import React, { useState, useEffect } from 'react';
import {
  View,
  Text,
  StyleSheet,
  FlatList,
  TouchableOpacity,
  SafeAreaView,
  ActivityIndicator,
  Alert,
  Image,
} from 'react-native';
import { Ionicons } from '@expo/vector-icons';

interface Highlight {
  id: string;
  player_number: string;
  player_name: string;
  team: string;
  ice_time: number;
  shifts: number;
  video_path: string;
  duration: number;
  thumbnail?: string;
}

const GameAnalysisScreen = ({ navigation }: any) => {
  const [highlights, setHighlights] = useState<Highlight[]>([]);
  const [isLoading, setIsLoading] = useState(false);
  const [selectedGame, setSelectedGame] = useState('game1');

  // 더미 데이터
  const dummyHighlights: Highlight[] = [
    {
      id: '1',
      player_number: '4',
      player_name: '윤지성',
      team: 'Aigis',
      ice_time: 15.5,
      shifts: 12,
      video_path: 'file:///path/to/highlight_4.mp4',
      duration: 930,
    },
    {
      id: '2',
      player_number: '11',
      player_name: '박리오',
      team: 'Aigis',
      ice_time: 18.2,
      shifts: 14,
      video_path: 'file:///path/to/highlight_11.mp4',
      duration: 1092,
    },
    {
      id: '3',
      player_number: '14',
      player_name: '이봄',
      team: 'Aigis',
      ice_time: 16.8,
      shifts: 13,
      video_path: 'file:///path/to/highlight_14.mp4',
      duration: 1008,
    },
  ];

  useEffect(() => {
    loadHighlights();
  }, [selectedGame]);

  const loadHighlights = async () => {
    setIsLoading(true);
    try {
      // 실제로는 서버에서 가져오기
      // const response = await fetch(`https://api.iceiq.app/highlights/${selectedGame}`);
      // const data = await response.json();
      // setHighlights(data);

      // 더미 데이터 사용
      setHighlights(dummyHighlights);
    } catch (error) {
      console.error('Error loading highlights:', error);
      Alert.alert('오류', '하이라이트를 불러올 수 없습니다.');
    } finally {
      setIsLoading(false);
    }
  };

  const renderHighlightCard = ({ item }: { item: Highlight }) => {
    return (
      <TouchableOpacity
        style={styles.card}
        onPress={() =>
          navigation.navigate('HighlightShare', {
            video_path: item.video_path,
            player_number: item.player_number,
            player_name: item.player_name,
            team: item.team,
            video_stem: selectedGame,
            duration: item.duration,
          })
        }
      >
        {/* 썸네일 */}
        <View style={styles.thumbnailContainer}>
          <View style={styles.placeholderThumbnail}>
            <Ionicons name="videocam" size={40} color="#ddd" />
          </View>

          {/* 숫자 배지 */}
          <View style={styles.numberBadge}>
            <Text style={styles.numberText}>#{item.player_number}</Text>
          </View>

          {/* 공유 버튼 (우상단) */}
          <TouchableOpacity
            style={styles.quickShareButton}
            onPress={(e) => {
              e.stopPropagation();
              navigation.navigate('HighlightShare', {
                video_path: item.video_path,
                player_number: item.player_number,
                player_name: item.player_name,
                team: item.team,
                video_stem: selectedGame,
                duration: item.duration,
              });
            }}
          >
            <Ionicons name="share-social" size={18} color="#fff" />
          </TouchableOpacity>
        </View>

        {/* 정보 */}
        <View style={styles.cardContent}>
          <View style={styles.playerHeader}>
            <Text style={styles.playerName}>{item.player_name}</Text>
            <Text style={styles.team}>{item.team}</Text>
          </View>

          {/* 통계 */}
          <View style={styles.stats}>
            <View style={styles.statItem}>
              <Ionicons name="time" size={14} color="#666" />
              <Text style={styles.statText}>
                {Math.floor(item.ice_time)}:{(Math.round((item.ice_time % 1) * 60)).toString().padStart(2, '0')}
              </Text>
            </View>
            <View style={styles.statItem}>
              <Ionicons name="repeat" size={14} color="#666" />
              <Text style={styles.statText}>{item.shifts}회</Text>
            </View>
          </View>

          {/* 하단 버튼 */}
          <TouchableOpacity
            style={styles.playButton}
            onPress={() =>
              navigation.navigate('HighlightShare', {
                video_path: item.video_path,
                player_number: item.player_number,
                player_name: item.player_name,
                team: item.team,
                video_stem: selectedGame,
                duration: item.duration,
              })
            }
          >
            <Ionicons name="play-circle" size={16} color="#1f77d2" />
            <Text style={styles.playButtonText}>재생 & 공유</Text>
          </TouchableOpacity>
        </View>
      </TouchableOpacity>
    );
  };

  return (
    <SafeAreaView style={styles.container}>
      {/* 헤더 */}
      <View style={styles.header}>
        <Text style={styles.headerTitle}>경기 분석</Text>
        <Ionicons name="analytics" size={24} color="#1f77d2" />
      </View>

      {/* 경기 선택 */}
      <View style={styles.gameSelector}>
        <TouchableOpacity
          style={[styles.gameTab, selectedGame === 'game1' && styles.activeGameTab]}
          onPress={() => setSelectedGame('game1')}
        >
          <Text
            style={[styles.gameTabText, selectedGame === 'game1' && styles.activeGameTabText]}
          >
            경기 1
          </Text>
        </TouchableOpacity>
        <TouchableOpacity
          style={[styles.gameTab, selectedGame === 'game2' && styles.activeGameTab]}
          onPress={() => setSelectedGame('game2')}
        >
          <Text
            style={[styles.gameTabText, selectedGame === 'game2' && styles.activeGameTabText]}
          >
            경기 2
          </Text>
        </TouchableOpacity>
        <TouchableOpacity
          style={[styles.gameTab, selectedGame === 'game3' && styles.activeGameTab]}
          onPress={() => setSelectedGame('game3')}
        >
          <Text
            style={[styles.gameTabText, selectedGame === 'game3' && styles.activeGameTabText]}
          >
            경기 3
          </Text>
        </TouchableOpacity>
      </View>

      {/* 하이라이트 목록 */}
      {isLoading ? (
        <View style={styles.loadingContainer}>
          <ActivityIndicator size="large" color="#1f77d2" />
          <Text style={styles.loadingText}>로딩 중...</Text>
        </View>
      ) : (
        <FlatList
          data={highlights}
          renderItem={renderHighlightCard}
          keyExtractor={(item) => item.id}
          contentContainerStyle={styles.listContainer}
          scrollEventThrottle={16}
          showsVerticalScrollIndicator={false}
        />
      )}

      {/* 빈 상태 */}
      {!isLoading && highlights.length === 0 && (
        <View style={styles.emptyContainer}>
          <Ionicons name="videocam-off" size={48} color="#ddd" />
          <Text style={styles.emptyText}>하이라이트가 없습니다</Text>
        </View>
      )}
    </SafeAreaView>
  );
};

const styles = StyleSheet.create({
  container: {
    flex: 1,
    backgroundColor: '#f5f5f5',
  },
  header: {
    flexDirection: 'row',
    justifyContent: 'space-between',
    alignItems: 'center',
    paddingHorizontal: 16,
    paddingVertical: 12,
    backgroundColor: '#fff',
    borderBottomWidth: 1,
    borderBottomColor: '#eee',
  },
  headerTitle: {
    fontSize: 20,
    fontWeight: '700',
    color: '#333',
  },
  gameSelector: {
    flexDirection: 'row',
    paddingHorizontal: 16,
    paddingVertical: 12,
    gap: 8,
    backgroundColor: '#fff',
  },
  gameTab: {
    flex: 1,
    paddingVertical: 8,
    paddingHorizontal: 12,
    borderRadius: 6,
    backgroundColor: '#f0f0f0',
    alignItems: 'center',
  },
  activeGameTab: {
    backgroundColor: '#1f77d2',
  },
  gameTabText: {
    fontSize: 13,
    fontWeight: '600',
    color: '#666',
  },
  activeGameTabText: {
    color: '#fff',
  },
  listContainer: {
    paddingHorizontal: 16,
    paddingVertical: 12,
    gap: 12,
  },
  card: {
    flexDirection: 'row',
    backgroundColor: '#fff',
    borderRadius: 8,
    overflow: 'hidden',
    borderWidth: 1,
    borderColor: '#eee',
  },
  thumbnailContainer: {
    position: 'relative',
    width: 120,
    height: 120,
    backgroundColor: '#f0f0f0',
    justifyContent: 'center',
    alignItems: 'center',
  },
  placeholderThumbnail: {
    justifyContent: 'center',
    alignItems: 'center',
  },
  numberBadge: {
    position: 'absolute',
    bottom: 8,
    left: 8,
    backgroundColor: '#1f77d2',
    paddingVertical: 4,
    paddingHorizontal: 8,
    borderRadius: 4,
  },
  numberText: {
    fontSize: 14,
    fontWeight: '700',
    color: '#fff',
  },
  quickShareButton: {
    position: 'absolute',
    top: 8,
    right: 8,
    backgroundColor: 'rgba(31, 119, 210, 0.9)',
    width: 32,
    height: 32,
    borderRadius: 16,
    justifyContent: 'center',
    alignItems: 'center',
  },
  cardContent: {
    flex: 1,
    paddingHorizontal: 12,
    paddingVertical: 8,
    justifyContent: 'space-between',
  },
  playerHeader: {
    marginBottom: 8,
  },
  playerName: {
    fontSize: 16,
    fontWeight: '700',
    color: '#333',
    marginBottom: 2,
  },
  team: {
    fontSize: 12,
    color: '#999',
  },
  stats: {
    flexDirection: 'row',
    gap: 12,
    marginBottom: 8,
  },
  statItem: {
    flexDirection: 'row',
    alignItems: 'center',
    gap: 4,
  },
  statText: {
    fontSize: 12,
    color: '#666',
  },
  playButton: {
    flexDirection: 'row',
    alignItems: 'center',
    justifyContent: 'center',
    gap: 6,
    paddingVertical: 6,
    paddingHorizontal: 8,
    backgroundColor: '#f0f0f0',
    borderRadius: 4,
  },
  playButtonText: {
    fontSize: 12,
    fontWeight: '600',
    color: '#1f77d2',
  },
  loadingContainer: {
    flex: 1,
    justifyContent: 'center',
    alignItems: 'center',
  },
  loadingText: {
    marginTop: 12,
    fontSize: 14,
    color: '#999',
  },
  emptyContainer: {
    flex: 1,
    justifyContent: 'center',
    alignItems: 'center',
  },
  emptyText: {
    marginTop: 12,
    fontSize: 14,
    color: '#999',
  },
});

export default GameAnalysisScreen;
