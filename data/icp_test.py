import numpy as np
import time
import icp

# Constants
N = 10  # 데이터셋 크기
num_tests = 10  # 반복 테스트 계산 횟수
dim = 3  # 데이터 포인트 차원
noise_sigma = .01  # 노이즈 표준 편차
translation = .1  # 테스트셋 최대 이동 거리
rotation = .1  # 테스트셋 최대 회전 각

def rotation_matrix(axis, theta):
    axis = axis/np.sqrt(np.dot(axis, axis))
    a = np.cos(theta/2.)
    b, c, d = -axis*np.sin(theta/2.)

    return np.array([[a*a+b*b-c*c-d*d, 2*(b*c-a*d), 2*(b*d+a*c)],
                  [2*(b*c+a*d), a*a+c*c-b*b-d*d, 2*(c*d-a*b)],
                  [2*(b*d-a*c), 2*(c*d+a*b), a*a+d*d-b*b-c*c]])


def test_best_fit():
    # Generate a random dataset
    A = np.random.rand(N, dim) #(N,3)

    total_time = 0

    for i in range(num_tests):

        B = np.copy(A)

        # Translate
        t = np.random.rand(dim)*translation
        B += t

        # Rotate
        R = rotation_matrix(np.random.rand(dim), np.random.rand()*rotation) #(3,3)
        B = np.dot(R, B.T).T #(N,3)

        # Add noise
        B += np.random.randn(N, dim) * noise_sigma

        # Find best fit transform
        start = time.time()
        T, R1, t1 = icp.best_fit_transform(B, A)
        total_time += time.time() - start

        # Make C a homogeneous representation of B
        C = np.ones((N, 4))
        C[:,0:3] = B

        # Transform C
        C = np.dot(T, C.T).T

        assert np.allclose(C[:,0:3], A, atol=6*noise_sigma) # T should transform B (or C) to A
        assert np.allclose(-t1, t, atol=6*noise_sigma)      # t and t1 should be inverses
        assert np.allclose(R1.T, R, atol=6*noise_sigma)     # R and R1 should be inverses

    print('best fit time: {:.3}'.format(total_time/num_tests))

    return


def test_icp():
    # 임의 데이터셋 생성
    A = np.random.rand(N, dim)

    total_time = 0
    final_RT = []
    for i in range(num_tests):
        B = np.copy(A)

        # 테스트 데이터셋 이동
        t = np.random.rand(dim) * translation
        B += t

        # 회전
        R = rotation_matrix(np.random.rand(dim), np.random.rand() * rotation)
        B = np.dot(R, B.T).T

        # 노이즈 추가
        B += np.random.randn(N, dim) * noise_sigma

        # 위치 섞음
        np.random.shuffle(B)

        # ICP 알고리즘 실행
        start = time.time()
        T, distances, iterations = icp.icp(B, A, tolerance=0.000001) #T(4,4)
        final_RT = T
        total_time += time.time() - start

        # 동차좌표 생성
        C = np.ones((N, 4))
        C[:, 0:3] = np.copy(B)

        # 변환행렬 적용
        C = np.dot(T, C.T).T

        print('distance: {:.3}'.format(np.mean(distances)))

        assert np.mean(distances) < 6 * noise_sigma  # 평균 에러
        assert np.allclose(T[0:3, 0:3].T, R, atol=6 * noise_sigma)  # T and R should be inverses
        assert np.allclose(-T[0:3, 3], t, atol=6 * noise_sigma)  # T and t should be inverses

    #RT(4,4) -> (1,12) txt file
    print('icp time: {:.3}'.format(total_time / num_tests))

    final_RT = np.reshape(final_RT,(1,-1))


    return



if __name__ == "__main__":
    test_best_fit()
    test_icp()