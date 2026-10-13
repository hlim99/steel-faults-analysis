import argparse
from pathlib import Path

import matplotlib.pyplot as plt
import pandas as pd
from sklearn.dummy import DummyClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler

'''열 (Column, 세로)→↓'''
'''행 (Row, 가로)↓→'''
'''FEATURE_COLS 특징이면 TARGET_COLS 결함이다 -> 지도학습'''
'''비지도학습은 정답 없음(비슷한 것끼리 묶기), 강화학습은 보상만 있음(잘하면 플러스, 못하면 마이너스로 알아서 배우게 하기)'''

# 컬럼명 (Faults.NNA 파일에는 헤더가 없어서 직접 지정)
FEATURE_COLS = [
    "X_Minimum", "X_Maximum", "Y_Minimum", "Y_Maximum", "Pixels_Areas",
    "X_Perimeter", "Y_Perimeter", "Sum_of_Luminosity", "Minimum_of_Luminosity",
    "Maximum_of_Luminosity", "Length_of_Conveyer", "TypeOfSteel_A300",
    "TypeOfSteel_A400", "Steel_Plate_Thickness", "Edges_Index", "Empty_Index",
    "Square_Index", "Outside_X_Index", "Edges_X_Index", "Edges_Y_Index",
    "Outside_Global_Index", "LogOfAreas", "Log_X_Index", "Log_Y_Index",
    "Orientation_Index", "Luminosity_Index", "SigmoidOfAreas",
]
TARGET_COLS = ["Pastry", "Z_Scratch", "K_Scatch", "Stains", "Dirtiness", "Bumps", "Other_Faults"]


def load_data(csv_path: str) -> pd.DataFrame:
    path = Path(csv_path)
    if not path.exists():
        raise FileNotFoundError(f"파일을 찾을 수 없습니다: {csv_path}")
    '''paht를 읽어올거임 / pandas는, 즉 pd는 첫줄부터 컬럼명으로 사용함 그래서 header = None 으로 첫줄도 데이터로 취급하라는 의미 / .NNA 파일이 탭으로 구분되어 있으니까 탭으로 구분하라는 의미'''
    df = pd.read_csv(path, header=None, sep="\t")
    df.columns = FEATURE_COLS + TARGET_COLS
    return df


def check_data(df: pd.DataFrame):
    """0단계: 데이터 구조 확인, 결측치 확인, 클래스 분포 확인"""
    print("=" * 60)
    print("[0단계] 데이터 확인")
    print("=" * 60)

    # 1. 데이터 구조
    print(f"\n[데이터 구조]")
    '''shape은 속성 shape[행, 열] 이라서 행은 0 열은 1'''
    print(f"  행(샘플 수): {df.shape[0]}") 
    print(f"  열(변수 수): {df.shape[1]}  (특성 {len(FEATURE_COLS)}개 + 타깃 {len(TARGET_COLS)}개)")

    # 2. 결측치 확인
    print(f"\n[결측치 확인]")
    '''missing은 컬럼별 결측치 개수'''
    missing = df.isnull().sum()
    '''missing_cols에 컬럼 결측치 이름+개수 Series(판다스 객체)'''
    '''컬럼만 보고 싶을 때: missing_cols.index.tolist()'''
    missing_cols = missing[missing > 0]
    if missing_cols.empty:
        print("  결측치 없음")
    else:
        for col, cnt in missing_cols.items():
            print(f"  {col}: {cnt}개")

    # 3. 클래스 분포 확인
    print(f"\n[클래스 분포]")
    '''df[TARGET_COLS은 7].각 행 기준으로 제일 큰 값(=1)이 있는 컬럼 이름 반환'''
    '''참고: axis=0(열 기준) / axis=1(행 기준)'''
    '''여기가 원 핫 인코딩(각 행에 1이 딱 하나 있을 때만 안전한 방법, 즉 전부 0 또는 1이 두개 이상이면 첫번째 칼럼 반환 = 오류)'''
    y_temp = df[TARGET_COLS].idxmax(axis=1)
    '''value_counts: 같은 값 몇번 나왔는지 세기 / 즉, TARGET_COLS에 원핫으로 걸린 애들 누가 몇번 걸렸나 확인'''
    '''ex) Pastry       2
           K_Scratch    2
           Z_Scratch    1'''
    counts = y_temp.value_counts()
    '''y_temp는 pandas의 Series이고 행의 길이'''
    total = len(y_temp)
    '''Pastry       10
       K_Scratch     5
       Z_Scratch     3
       이건 Series 즉 count 변수에 저장된 값
       .items() 역할은 데이터를 바꿔줌:
       ("Pastry", 10)
       ("K_Scratch", 5)
       ("Z_Scratch", 3)

       label → 결함 이름 (Pastry)
       cnt → 개수 (10)'''
    for label, cnt in counts.items():
        # ASCII 문자를 사용해 CP949 기반 Windows 콘솔에서도 출력 오류를 방지한다.
        bar = "#" * int(cnt / total * 30)
        '''cnt/total = 특정 결함 개수 / 전체 데이터 개수'''
        print(f"  {label:<15} {cnt:>4}개  ({cnt/total*100:5.1f}%)  {bar}")

    # 4. 변수 분포 확인 (기초 통계)
    print(f"[변수 분포 - 기초 통계]")
    stats = df[FEATURE_COLS].describe().T[["min", "max", "mean", "std"]]
    '''컬럼 선택 → 통계 계산(describe()) → 뒤집기(T) → 필요한 것만 추리기(min max..)'''
    '''        f1     f2     f3
        count  100    100    100
        mean    10     20     30
        std      2      3      4
        min      5     10     15
        max     15     30     45
이거에서
        count  mean  std  min  max
f1        100    10    2    5    15
f2        100    20    3   10    30
f3        100    30    4   15    45
이거로 바뀜'''
    stats["range"] = stats["max"] - stats["min"]
    '''      min   max   mean   std   range
        f1     5    15     10     2     10
        f2    10    30     20     3     20
        f3    15    45     30     4     30
        “이 변수 값이 얼마나 넓게 퍼져있냐” 보는 지표'''
    print(f"  {'변수명':<25} {'최솟값':>10} {'최댓값':>10} {'평균':>10} {'표준편차':>10} {'범위':>10}")
    print("  " + "-" * 75)
    for col, row in stats.iterrows():
        '''iterrows(): 표에서 각각 한줄씩(한 행 즉, f1 min max보고 f2 min max 보고 쉽게 볼 수 있게) 보고 갖어오는거임'''
        print(f"  {col:<25} {row['min']:>10.2f} {row['max']:>10.2f} {row['mean']:>10.2f} {row['std']:>10.2f} {row['range']:>10.2f}")

    print()


def prepare_data(df: pd.DataFrame):
    X = df[FEATURE_COLS]
    y = df[TARGET_COLS].idxmax(axis=1)
    '''원핫인코딩 axis=1 이니까 행'''
    return X, y
    '''x는 문제 y는 정답을 반환'''


def run_base_comparison(X_train, X_test, y_train, y_test):
    """기존 코드: Dummy / RF 기본 / RF 스케일링 비교"""
    print("=" * 60)
    print("[1단계] 기본 모델 비교")
    print("=" * 60)

    # Dummy Classifier
    dummy = DummyClassifier(strategy="most_frequent")
    dummy.fit(X_train, y_train)
    dummy_acc = accuracy_score(y_test, dummy.predict(X_test))
    print(f"Dummy Classifier         테스트 정확도: {dummy_acc:.4f}")

    # Random Forest 기본
    rf_base = RandomForestClassifier(n_estimators=200, random_state=42, class_weight="balanced")
    rf_base.fit(X_train, y_train)
    base_train = accuracy_score(y_train, rf_base.predict(X_train))
    base_test  = accuracy_score(y_test,  rf_base.predict(X_test))
    print(f"RF 기본      훈련: {base_train:.4f} | 테스트: {base_test:.4f} | 차이: {base_train - base_test:.4f}")

    # Random Forest 스케일링
    scaler = StandardScaler()
    X_tr_s = scaler.fit_transform(X_train)
    X_te_s = scaler.transform(X_test)
    rf_scaled = RandomForestClassifier(n_estimators=200, random_state=42, class_weight="balanced")
    rf_scaled.fit(X_tr_s, y_train)
    scaled_train = accuracy_score(y_train, rf_scaled.predict(X_tr_s))
    scaled_test  = accuracy_score(y_test,  rf_scaled.predict(X_te_s))
    print(f"RF 스케일링  훈련: {scaled_train:.4f} | 테스트: {scaled_test:.4f} | 차이: {scaled_train - scaled_test:.4f}")

    return rf_base, dummy_acc, base_train, base_test, scaled_test


def run_depth_comparison(X_train, X_test, y_train, y_test):
    """추가 실험 1: max_depth 변화에 따른 과적합 변화"""
    print("\n" + "=" * 60)
    print("[2단계] 트리 깊이 제한(max_depth) 실험")
    print("=" * 60)
    print(f"{'max_depth':>12} | {'훈련':>8} | {'테스트':>8} | {'차이':>8}")
    print("-" * 48)

    depth_results = []
    for depth in [None, 3, 5, 10, 15, 20]:
        rf = RandomForestClassifier(n_estimators=200, max_depth=depth, random_state=42, class_weight="balanced")
        rf.fit(X_train, y_train)
        tr = accuracy_score(y_train, rf.predict(X_train))
        te = accuracy_score(y_test,  rf.predict(X_test))
        label = "None\n(base)" if depth is None else str(depth)
        print(f"{label:>12} | {tr:>8.4f} | {te:>8.4f} | {tr - te:>8.4f}")
        depth_results.append((label, tr, te))

    return depth_results


def run_leaf_comparison(X_train, X_test, y_train, y_test):
    """추가 실험 2: min_samples_leaf 변화에 따른 과적합 변화"""
    print("\n" + "=" * 60)
    print("[3단계] 최소 샘플 수 제한(min_samples_leaf) 실험")
    print("=" * 60)
    print(f"{'min_samples_leaf':>18} | {'훈련':>8} | {'테스트':>8} | {'차이':>8}")
    print("-" * 52)

    leaf_results = []
    for leaf in [1, 3, 5, 10, 20, 50]:
        rf = RandomForestClassifier(n_estimators=200, min_samples_leaf=leaf, random_state=42, class_weight="balanced")
        rf.fit(X_train, y_train)
        tr = accuracy_score(y_train, rf.predict(X_train))
        te = accuracy_score(y_test,  rf.predict(X_test))
        label = "1\n(base)" if leaf == 1 else str(leaf)
        print(f"{label:>18} | {tr:>8.4f} | {te:>8.4f} | {tr - te:>8.4f}")
        leaf_results.append((label, tr, te))

    return leaf_results


def save_eda_charts(df: pd.DataFrame, y: pd.Series, output_dir: str):
    """데이터 확인 차트: 클래스 분포 + 변수 분포(boxplot)"""
    Path(output_dir).mkdir(parents=True, exist_ok=True)

    fig, axes = plt.subplots(1, 2, figsize=(16, 6))
    fig.suptitle("Steel Plates Faults - Data Check", fontsize=16, fontweight="bold")

    # (1) 클래스 분포
    ax = axes[0]
    counts = y.value_counts()
    colors = ["#ef4444" if counts[c] < 100 else "#f59e0b" if counts[c] < 300 else "#2563eb"
              for c in counts.index]
    bars = ax.barh(counts.index, counts.values, color=colors)
    ax.set_title("Class Distribution")
    ax.set_xlabel("Count")
    for bar, cnt in zip(bars, counts.values):
        ax.text(bar.get_width() + 5, bar.get_y() + bar.get_height() / 2,
                str(cnt), va="center", fontsize=9)
    ax.set_xlim(0, counts.max() * 1.15)

    # (2) 변수 분포 (상위 10개 boxplot)
    ax = axes[1]
    top10_features = df[FEATURE_COLS].std().sort_values(ascending=False).head(10).index.tolist()
    df_norm = (df[top10_features] - df[top10_features].min()) / \
              (df[top10_features].max() - df[top10_features].min() + 1e-9)
    df_norm.boxplot(ax=ax, vert=False, patch_artist=True,
                    boxprops=dict(facecolor="#bfdbfe"),
                    medianprops=dict(color="#1d4ed8", linewidth=2))
    ax.set_title("Top 10 Features Distribution (Normalized)")
    ax.set_xlabel("Normalized Value (0~1)")
    ax.grid(axis="x", alpha=0.3)

    plt.tight_layout()
    plt.savefig(Path(output_dir) / "eda_result.png", dpi=200, bbox_inches="tight")
    plt.close()
    print(f"EDA 차트 저장 완료: {output_dir}/eda_result.png\n")


def save_charts(rf_base, X, dummy_acc, base_test, scaled_test,
                depth_results, leaf_results, output_dir):
    Path(output_dir).mkdir(parents=True, exist_ok=True)

    fig, axes = plt.subplots(2, 2, figsize=(16, 12))
    fig.suptitle("Steel Plates Faults - Model Analysis", fontsize=16, fontweight="bold")

    # (1) 기본 모델 정확도 비교
    ax = axes[0, 0]
    models = ["Dummy", "RF Basic", "RF Scaled"]
    accs   = [dummy_acc, base_test, scaled_test]
    colors = ["#94a3b8", "#f59e0b", "#10b981"]
    bars = ax.bar(models, accs, color=colors, width=0.5)
    ax.set_ylim(0, 1.05)
    ax.set_title("Model Test Accuracy Comparison")
    ax.set_ylabel("Accuracy")
    for bar, acc in zip(bars, accs):
        ax.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 0.01,
                f"{acc:.4f}", ha="center", fontsize=10, fontweight="bold")

    # (2) Feature Importance Top 10
    ax = axes[0, 1]
    importances = pd.Series(rf_base.feature_importances_, index=X.columns)
    top10 = importances.sort_values(ascending=False).head(10)
    top10.sort_values().plot(kind="barh", ax=ax, color="#2563eb")
    ax.set_title("Top 10 Feature Importances")
    ax.set_xlabel("Importance")

    # (3) max_depth 실험
    ax = axes[1, 0]
    labels = [r[0] for r in depth_results]
    trains = [r[1] for r in depth_results]
    tests  = [r[2] for r in depth_results]
    x = range(len(labels))
    ax.plot(x, trains, "o-", label="Train", color="#f59e0b", linewidth=2)
    ax.plot(x, tests,  "s-", label="Test",  color="#2563eb", linewidth=2)
    ax.set_xticks(x)
    ax.set_xticklabels(labels, fontsize=9)
    ax.set_ylim(0, 1.05)
    ax.set_title("Accuracy by max_depth")
    ax.set_xlabel("max_depth")
    ax.set_ylabel("Accuracy")
    ax.legend()
    ax.grid(axis="y", alpha=0.3)

    # (4) min_samples_leaf 실험
    ax = axes[1, 1]
    labels = [r[0] for r in leaf_results]
    trains = [r[1] for r in leaf_results]
    tests  = [r[2] for r in leaf_results]
    x = range(len(labels))
    ax.plot(x, trains, "o-", label="Train", color="#f59e0b", linewidth=2)
    ax.plot(x, tests,  "s-", label="Test",  color="#2563eb", linewidth=2)
    ax.set_xticks(x)
    ax.set_xticklabels(labels, fontsize=9)
    ax.set_ylim(0, 1.05)
    ax.set_title("Accuracy by min_samples_leaf")
    ax.set_xlabel("min_samples_leaf")
    ax.set_ylabel("Accuracy")
    ax.legend()
    ax.grid(axis="y", alpha=0.3)

    plt.tight_layout()
    plt.savefig(Path(output_dir) / "analysis_result.png", dpi=200, bbox_inches="tight")
    plt.close()
    print(f"\n차트 저장 완료: {output_dir}/analysis_result.png")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Steel Plates Faults 분류 분석")
    parser.add_argument("--csv", type=str, default="Faults.NNA")
    parser.add_argument("--output-dir", type=str, default="outputs")
    args = parser.parse_args()

    df = load_data(args.csv)
    check_data(df)
    X, y = prepare_data(df)

    save_eda_charts(df, y, args.output_dir)

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y
    )

    rf_base, dummy_acc, base_train, base_test, scaled_test = run_base_comparison(X_train, X_test, y_train, y_test)
    depth_results = run_depth_comparison(X_train, X_test, y_train, y_test)
    leaf_results  = run_leaf_comparison(X_train, X_test, y_train, y_test)

    save_charts(rf_base, X, dummy_acc, base_test, scaled_test,
                depth_results, leaf_results, args.output_dir)
