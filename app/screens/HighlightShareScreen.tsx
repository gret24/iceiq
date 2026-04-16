/**
 * IceIQ Highlight Share Screen
 * 
 * 하이라이트 재생 & 공유 기능
 * React Native (Expo / React Native CLI)
 */

import React, { useState, useRef } from 'react';
import {
  View,
  Text,
  StyleSheet,
  TouchableOpacity,
  Share,
  ActivityIndicator,
  Alert,
  Dimensions,
  SafeAreaView,
  Platform,
} from 'react-native';
import { Video } from 'expo-av';
import * as FileSystem from 'expo-file-system';
import { Ionicons } from '@expo/vector-icons';

const { width, height } = Dimensions.get('window');

interface HighlightData {
  video_path: string;
  player_number: string;
  player_name: string;
  team: string;
  video_stem: string;
  duration: number;
}

interface ShareScreenProps {
  route: {
    params: HighlightData;
  };
}

const HighlightShareScreen: React.FC<ShareScreenProps> = ({ route }) => {
  const { video_path, player_number, player_name, team, video_stem, duration } =
    route.params;

  const videoRef = useRef(null);
  const [isLoading, setIsLoading] = useState(false);
  const [isSharing, setIsSharing] = useState(false);
  const [shareUrl, setShareUrl] = useState<string | null>(null);

  /**
   * 1️⃣ 하이라이트 영상을 서버에 업로드 & 공유 URL 생성
   */
  const uploadAndShare = async () => {
    try {
      setIsLoading(true);

      // 로컬 영상 파일 정보 가져오기
      const fileInfo = await FileSystem.getInfoAsync(video_path);
      if (!fileInfo.exists) {
        Alert.alert('오류', '영상 파일을 찾을 수 없습니다.');
        return;
      }

      // FormData 생성
      const formData = new FormData();
      formData.append('file', {
        uri: video_path,
        type: 'video/mp4',
        name: `highlight_${player_number}.mp4`,
      } as any);
      formData.append('video_stem', video_stem);
      formData.append('player', player_number);

      // 서버에 업로드 (POST /share)
      const response = await fetch('https://api.iceiq.app/share', {
        method: 'POST',
        body: formData,
        headers: {
          Accept: 'application/json',
        },
      });

      if (!response.ok) {
        throw new Error(`Upload failed: ${response.status}`);
      }

      const data = await response.json();
      const { share_url, watermarked } = data;

      setShareUrl(share_url);

      // 성공 메시지
      Alert.alert(
        '공유 준비 완료!',
        `${watermarked ? '✅ 워터마크 적용됨\n' : ''}${share_url}`,
        [
          {
            text: '공유하기',
            onPress: () => onShare(share_url, player_name),
          },
          {
            text: '닫기',
            onPress: () => {},
          },
        ]
      );

      setIsLoading(false);
    } catch (error) {
      console.error('Upload error:', error);
      Alert.alert('업로드 실패', '영상을 업로드할 수 없습니다.');
      setIsLoading(false);
    }
  };

  /**
   * 2️⃣ OS 공유 시트 열기 (카톡, 인스타, 문자, 이메일)
   */
  const onShare = async (url: string, playerName: string) => {
    try {
      setIsSharing(true);

      const message = `${playerName} 선수의 하이라이트를 확인하세요!\n\n⚡ IceIQ로 분석된 경기 영상입니다.`;

      const result = await Share.share({
        message:
          Platform.OS === 'ios'
            ? message // iOS에서는 URL이 자동으로 추가됨
            : `${message}\n\n${url}`,
        url: url, // iOS에서만 동작
        title: 'IceIQ 하이라이트 공유',
      });

      if (result.action === Share.dismissedAction) {
        console.log('Share dismissed');
      }

      setIsSharing(false);
    } catch (error) {
      console.error('Share error:', error);
      Alert.alert('공유 실패', '공유할 수 없습니다.');
      setIsSharing(false);
    }
  };

  /**
   * 3️⃣ 공유 URL 직접 복사
   */
  const copyToClipboard = async () => {
    if (!shareUrl) {
      Alert.alert('알림', '먼저 공유하기를 준비해주세요.');
      return;
    }

    try {
      // React Native 클립보드
      const Clipboard = require('@react-native-clipboard/clipboard').default;
      Clipboard.setString(shareUrl);
      Alert.alert('복사됨', '공유 링크가 클립보드에 복사되었습니다.');
    } catch (error) {
      console.error('Clipboard error:', error);
    }
  };

  return (
    <SafeAreaView style={styles.container}>
      {/* 비디오 플레이어 */}
      <View style={styles.videoContainer}>
        <Video
          ref={videoRef}
          source={{ uri: video_path }}
          rate={1.0}
          volume={1.0}
          isMuted={false}
          resizeMode="contain"
          isLooping={false}
          shouldPlay={false}
          style={styles.video}
          useNativeControls
          progressUpdateIntervalMillis={500}
        />
      </View>

      {/* 선수 정보 */}
      <View style={styles.infoSection}>
        <View style={styles.playerInfo}>
          <Text style={styles.playerName}>{player_name}</Text>
          <Text style={styles.playerDetails}>
            #{player_number} • {team}
          </Text>
        </View>

        <View style={styles.metaInfo}>
          <View style={styles.metaItem}>
            <Ionicons name="videocam" size={16} color="#666" />
            <Text style={styles.metaText}>
              {Math.floor(duration / 60)}:{(duration % 60).toString().padStart(2, '0')}
            </Text>
          </View>
          <View style={styles.metaItem}>
            <Ionicons name="calendar" size={16} color="#666" />
            <Text style={styles.metaText}>{video_stem}</Text>
          </View>
        </View>
      </View>

      {/* 공유 URL (표시용) */}
      {shareUrl && (
        <View style={styles.urlSection}>
          <Text style={styles.urlLabel}>공유 링크</Text>
          <View style={styles.urlBox}>
            <Text style={styles.urlText} numberOfLines={1}>
              {shareUrl}
            </Text>
            <TouchableOpacity onPress={copyToClipboard} style={styles.copyButton}>
              <Ionicons name="copy" size={18} color="#1f77d2" />
            </TouchableOpacity>
          </View>
        </View>
      )}

      {/* 액션 버튼 */}
      <View style={styles.actionSection}>
        {/* 공유 준비 버튼 */}
        <TouchableOpacity
          style={[styles.button, styles.uploadButton]}
          onPress={uploadAndShare}
          disabled={isLoading}
        >
          {isLoading ? (
            <ActivityIndicator color="#fff" />
          ) : (
            <>
              <Ionicons name="cloud-upload" size={20} color="#fff" />
              <Text style={styles.buttonText}>공유 준비</Text>
            </>
          )}
        </TouchableOpacity>

        {/* 공유 버튼 (OS 공유 시트) */}
        <TouchableOpacity
          style={[styles.button, styles.shareButton, !shareUrl && styles.disabledButton]}
          onPress={() => shareUrl && onShare(shareUrl, player_name)}
          disabled={!shareUrl || isSharing}
        >
          {isSharing ? (
            <ActivityIndicator color="#1f77d2" />
          ) : (
            <>
              <Ionicons name="share-social" size={20} color="#1f77d2" />
              <Text style={styles.shareButtonText}>
                {shareUrl ? '공유하기' : '먼저 준비해주세요'}
              </Text>
            </>
          )}
        </TouchableOpacity>
      </View>

      {/* 안내 텍스트 */}
      <View style={styles.helpSection}>
        <Ionicons name="information-circle" size={16} color="#999" />
        <Text style={styles.helpText}>
          공유 준비를 누르면 워터마크가 적용되고{'\n'}
          공유 링크가 생성됩니다. (1-2분 소요)
        </Text>
      </View>
    </SafeAreaView>
  );
};

const styles = StyleSheet.create({
  container: {
    flex: 1,
    backgroundColor: '#f5f5f5',
  },
  videoContainer: {
    width: '100%',
    aspectRatio: 16 / 9,
    backgroundColor: '#000',
    justifyContent: 'center',
    alignItems: 'center',
  },
  video: {
    width: '100%',
    height: '100%',
  },
  infoSection: {
    padding: 16,
    backgroundColor: '#fff',
    borderBottomWidth: 1,
    borderBottomColor: '#eee',
  },
  playerInfo: {
    marginBottom: 12,
  },
  playerName: {
    fontSize: 18,
    fontWeight: '700',
    color: '#333',
    marginBottom: 4,
  },
  playerDetails: {
    fontSize: 14,
    color: '#666',
  },
  metaInfo: {
    flexDirection: 'row',
    gap: 16,
  },
  metaItem: {
    flexDirection: 'row',
    alignItems: 'center',
    gap: 6,
  },
  metaText: {
    fontSize: 13,
    color: '#999',
  },
  urlSection: {
    padding: 16,
    backgroundColor: '#f9f9f9',
  },
  urlLabel: {
    fontSize: 12,
    fontWeight: '600',
    color: '#999',
    marginBottom: 8,
    textTransform: 'uppercase',
  },
  urlBox: {
    flexDirection: 'row',
    alignItems: 'center',
    backgroundColor: '#fff',
    borderRadius: 8,
    paddingHorizontal: 12,
    paddingVertical: 10,
    borderWidth: 1,
    borderColor: '#ddd',
  },
  urlText: {
    flex: 1,
    fontSize: 13,
    color: '#1f77d2',
    fontFamily: 'Menlo',
  },
  copyButton: {
    padding: 8,
  },
  actionSection: {
    paddingHorizontal: 16,
    paddingVertical: 12,
    gap: 10,
  },
  button: {
    flexDirection: 'row',
    alignItems: 'center',
    justifyContent: 'center',
    paddingVertical: 14,
    borderRadius: 8,
    gap: 8,
  },
  uploadButton: {
    backgroundColor: '#1f77d2',
  },
  shareButton: {
    backgroundColor: '#f0f0f0',
    borderWidth: 1,
    borderColor: '#1f77d2',
  },
  disabledButton: {
    opacity: 0.5,
  },
  buttonText: {
    fontSize: 16,
    fontWeight: '600',
    color: '#fff',
  },
  shareButtonText: {
    fontSize: 16,
    fontWeight: '600',
    color: '#1f77d2',
  },
  helpSection: {
    flexDirection: 'row',
    paddingHorizontal: 16,
    paddingVertical: 12,
    gap: 8,
    marginTop: 'auto',
    marginBottom: 16,
  },
  helpText: {
    fontSize: 12,
    color: '#999',
    flex: 1,
    lineHeight: 16,
  },
});

export default HighlightShareScreen;
