package SHAPSecureBoost

import (
	"context"
	"errors"
	"fmt"
	"github.com/haikongyu/SHAPSecureBoost/hostGRPC"
	"math"
)

func CalLeftGH(ctx context.Context, client hostGRPC.HostServiceClient, treeId, nodeId int64, G, H []float64, samplesId []int64) ([]int64, []float64, []float64) {
	nodeInfo := &hostGRPC.NodeInfo{
		TreeId:    treeId,
		NodeId:    nodeId,
		G:         G,
		H:         H,
		SamplesId: samplesId,
	}
	leftInfo, err := client.CalLeftGH(ctx, nodeInfo)
	if err != nil {
		//logger.Println("calLeftGH fail", err)
		return nil, nil, nil
	}
	//logger.Println("get leftGH after cal leftGH", leftInfo.SplitId, leftInfo.GL, leftInfo.HL)
	return leftInfo.SplitId, leftInfo.GL, leftInfo.HL
}

func HostSave(ctx context.Context, client hostGRPC.HostServiceClient, treeId, nodeId, splitId int64) (int64, []int64, []int64) {
	hostSavingInfo := &hostGRPC.HostSavingInfo{
		TreeId:  treeId,
		NodeId:  nodeId,
		SplitId: splitId,
		//SamplesId: samplesId,
	}
	splitInfo, err := client.HostSave(ctx, hostSavingInfo)
	if err != nil {
		//logger.Println("hostSave fail", err)
	}
	//logger.Println("host save success", splitInfo.FeatureId, splitInfo.LeftSamplesId, splitInfo.RightSamplesId)
	return splitInfo.FeatureId, splitInfo.LeftSamplesId, splitInfo.RightSamplesId
}

func FindNextNode(ctx context.Context, client hostGRPC.HostServiceClient, treeId, nodeId int64, samplesId []int64) []bool {
	nodeInfo := &hostGRPC.NodeInfo{
		TreeId:    treeId,
		NodeId:    nodeId,
		SamplesId: samplesId,
	}
	nextNode, err := client.FindNextNode(ctx, nodeInfo)
	if err != nil {
		//logger.Println("find next node fail", nextNode.IsLeft)
	}
	//logger.Println("next node", nextNode.IsLeft)
	return nextNode.IsLeft
}

type Node struct {
	featureOwner     string
	featureIndex     int64
	featureThreshold float64
	isLeaf           bool
	//left             *Node
	//right            *Node
	weight    float64
	depth     int64
	nodeIndex int64
	samplesId []int64
}

type Booster struct {
	id       int64
	maxDepth int64
	// maxDepth 包括根节点，不包含叶子节点
	lambda        float64
	gamma         float64
	g             []float64
	h             []float64
	tree          map[[2]int64]Node
	client        hostGRPC.HostServiceClient
	ctx           context.Context
	nHostFeatures int64
	learningRate  float64
}

func (booster *Booster) fit(X [][]float64, y []int64, score []float64) {
	booster.tree = make(map[[2]int64]Node)
	if len(y) != len(score) {
		panic(errors.New("在booster.fit 中 y, score 长度不一"))
	}
	p := Sigmoid(score)
	g := make([]float64, len(y))
	h := make([]float64, len(y))

	for i := 0; i < len(y); i++ {
		g[i] = p[i] - float64(y[i])
		h[i] = p[i] * (1 - p[i])
	}
	booster.g = g
	booster.h = h
	booster.growingTree(X, y, 0)
}

func (booster Booster) predict(X [][]float64) []float64 {
	weight := make([]float64, len(X))
	nodeLeftDict := map[int64][]bool{}
	samplesId := make([]int64, len(X))
	for i := int64(0); i < int64(len(X)); i++ {
		samplesId[i] = i
	}
	for i := 0; i < len(X); i++ {
		tree := booster.tree
		node := tree[[2]int64{0, 0}]
		for {
			if node.isLeaf {
				weight[i] = node.weight
				break
			} else if node.featureOwner == "guest" {
				if X[i][node.featureIndex] <= node.featureThreshold {
					node = tree[[2]int64{node.depth + 1, 2 * node.nodeIndex}]
				} else {
					node = tree[[2]int64{node.depth + 1, 2*node.nodeIndex + 1}]
				}
			} else if node.featureOwner == "host" {
				//批量查询减少通信次数和时间
				nodeId := int64(int(math.Exp2(float64(node.depth)))) + node.nodeIndex - 1
				if len(nodeLeftDict[nodeId]) == 0 {
					nodeLeftDict[nodeId] = FindNextNode(booster.ctx, booster.client, booster.id, nodeId, samplesId)
				}
				isLeft := nodeLeftDict[nodeId]
				if isLeft[i] {
					node = tree[[2]int64{node.depth + 1, 2 * node.nodeIndex}]
				} else {
					node = tree[[2]int64{node.depth + 1, 2*node.nodeIndex + 1}]
				}
				//isLeft := FindNextNode(booster.ctx, booster.client, booster.id, int64(int(math.Exp2(float64(node.depth))))+node.nodeIndex-1, []int64{int64(i)})
				//if isLeft[0] {
				//	node = tree[[2]int64{node.depth + 1, 2 * node.nodeIndex}]
				//} else {
				//	node = tree[[2]int64{node.depth + 1, 2*node.nodeIndex + 1}]
				//}
			}
		}
	}
	return weight
}

func (booster Booster) findBestSplit(X [][]float64, samplesId []int64, g, h []float64, depth, nodeIndex int64) (string, int64, float64, []int64, []int64) {
	//tcp
	gamma := booster.gamma
	lambda := booster.lambda
	if len(samplesId) <= 1 {
		return "guest", -1, -1, samplesId, []int64{}
	}
	G := 0.0
	H := 0.0
	bestGain := -booster.gamma
	for i := 0; i < len(samplesId); i++ {
		G += g[samplesId[i]]
		H += h[samplesId[i]]
	}
	bestIndex := int64(-1)
	bestThreshold := 0.0
	for i := int64(0); i < int64(len(X[0])); i++ {
		data := make([][]float64, len(samplesId))
		for j := range data {
			data[j] = make([]float64, 3)
		}
		var thresholds []float64
		m := map[float64]bool{}
		for j := 0; j < len(samplesId); j++ {
			data[j][0] = X[samplesId[j]][i]
			if _, ok := m[data[j][0]]; !ok {
				thresholds = append(thresholds, data[j][0])
				m[data[j][0]] = true
			}
			data[j][1] = g[samplesId[j]]
			data[j][2] = h[samplesId[j]]
		}
		for _, threshold := range thresholds {
			gl := 0.0
			hl := 0.0
			counter := 0
			for j := 0; j < len(samplesId); j++ {
				if data[j][0] <= threshold {
					gl += data[j][1]
					hl += data[j][2]
				} else {
					counter++
				}
			}
			gr := G - gl
			hr := H - hl
			gain := 0.5*(gl*gl/(hl+lambda)+gr*gr/(hr+lambda)-G*G/(H+lambda)) - gamma
			if gain > bestGain {
				bestGain = gain
				bestIndex = i
				bestThreshold = threshold
			}
		}
	}
	if bestGain <= 0 {
		bestIndex = -1
	}
	hostIndex, hostGL, hostHL := CalLeftGH(booster.ctx, booster.client, booster.id, int64(int(math.Exp2(float64(depth))))+nodeIndex-1, g, h, samplesId)
	hostBestIndex := int64(-1)
	hostBestGain := float64(0)
	for idx := int64(0); idx < int64(len(hostIndex)); idx++ {
		gl, hl := hostGL[idx], hostHL[idx]
		gr := G - gl
		hr := H - hl
		gain := 0.5*(gl*gl/(hl+lambda)+gr*gr/(hr+lambda)-G*G/(H+lambda)) - gamma
		if gain > hostBestGain {
			hostBestIndex = idx
			hostBestGain = gain
		}
	}
	if bestGain >= hostBestGain {
		if bestIndex == -1 {
			return "guest", bestIndex, bestThreshold, samplesId, []int64{}
		} else {
			leftSamplesId, rightSamplesId := make([]int64, 0, len(samplesId)), make([]int64, 0, len(samplesId))
			for i := range samplesId {
				if X[samplesId[i]][bestIndex] <= bestThreshold {
					leftSamplesId = append(leftSamplesId, samplesId[i])
				} else {
					rightSamplesId = append(rightSamplesId, samplesId[i])
				}
			}
			return "guest", bestIndex, bestThreshold, leftSamplesId, rightSamplesId
		}
	} else {
		featureId, leftSamplesId, rightSamplesId := HostSave(booster.ctx, booster.client, booster.id, int64(int(math.Exp2(float64(depth))))+nodeIndex-1, hostBestIndex)
		return "host", featureId, 0.0, leftSamplesId, rightSamplesId
	}
}

func (booster *Booster) growingTree(X [][]float64, y []int64, depth int64) {
	node := Node{featureOwner: "guest", depth: 0, nodeIndex: 0}
	fmt.Printf("第%v颗树，第%v层%v个节点训练中", booster.id, node.depth, node.nodeIndex)
	samplesId := make([]int64, len(X))
	for i := int64(0); i < int64(len(X)); i++ {
		samplesId[i] = i
	}
	node.samplesId = samplesId
	g := booster.g
	h := booster.h
	lambda := booster.lambda
	if depth < booster.maxDepth {
		owner, index, threshold, leftSamplesId, rightSamplesId := booster.findBestSplit(X, samplesId, g, h, 0, 0)
		if len(leftSamplesId)*len(rightSamplesId) == 0 {
			//index = -1
		}
		if index != -1 {
			node.featureOwner = owner
			node.featureIndex = index
			node.featureThreshold = threshold
			//左支生长
			booster.growingTree_(X, y, leftSamplesId, g, h, 1, 0)
			//右支生长
			booster.growingTree_(X, y, rightSamplesId, g, h, 1, 1)
		} else {
			node.isLeaf = true
			node.featureIndex = -1
		}

	} else {
		node.isLeaf = true
		node.featureIndex = -1
	}
	if node.isLeaf {
		G := 0.0
		H := 0.0
		for i := 0; i < len(g); i++ {
			G += g[i]
			H += h[i]
		}
		node.weight = -G / (H + lambda)
	}
	booster.tree[[2]int64{0, 0}] = node
	fmt.Println(booster.id, node.depth, node.nodeIndex, node.featureOwner, node.featureIndex, node.isLeaf)
}

func (booster *Booster) growingTree_(X [][]float64, y, samplesId []int64, g, h []float64, depth, nodeIndex int64) {
	node := Node{featureOwner: "guest", depth: depth, nodeIndex: nodeIndex}
	fmt.Printf("第%v颗树，第%v层%v个节点训练中", booster.id, node.depth, node.nodeIndex)
	node.samplesId = samplesId
	lambda := booster.lambda
	if depth < booster.maxDepth {
		owner, index, threshold, leftSamplesId, rightSamplesId := booster.findBestSplit(X, samplesId, g, h, depth, nodeIndex)
		if len(leftSamplesId)*len(rightSamplesId) == 0 {
			//index = -1
		}
		if index != -1 {
			node.featureOwner = owner
			node.featureIndex = index
			node.featureThreshold = threshold
			//左支生长
			booster.growingTree_(X, y, leftSamplesId, g, h, depth+1, 2*nodeIndex)
			//右支生长
			booster.growingTree_(X, y, rightSamplesId, g, h, depth+1, 2*nodeIndex+1)
		} else {
			node.isLeaf = true
			node.featureIndex = -1
		}
	} else {
		node.isLeaf = true
		node.featureIndex = -1
	}
	if node.isLeaf {
		G := 0.0
		H := 0.0
		for i := 0; i < len(samplesId); i++ {
			G += g[samplesId[i]]
			H += h[samplesId[i]]
		}
		node.weight = -G / (H + lambda)
	}
	booster.tree[[2]int64{depth, nodeIndex}] = node
	fmt.Println(booster.id, node.depth, node.nodeIndex, node.featureOwner, node.featureIndex, node.isLeaf)
}

func (booster Booster) TreeShapSubset(X [][]float64, sampleId int64, subsetIndex []int64, depth, nodeIndex int64) float64 {
	XI := X[sampleId]
	node := booster.tree[[2]int64{depth, nodeIndex}]
	if node.isLeaf {
		return node.weight * booster.learningRate
	} else {
		if node.featureOwner == "guest" {
			if subsetIndex[node.featureIndex] == 1 {
				if XI[node.featureIndex] <= node.featureThreshold {
					return booster.TreeShapSubset(X, sampleId, subsetIndex, depth+1, 2*nodeIndex)
				} else {
					return booster.TreeShapSubset(X, sampleId, subsetIndex, depth+1, 2*nodeIndex+1)
				}
			} else {
				left := booster.TreeShapSubset(X, sampleId, subsetIndex, depth+1, 2*nodeIndex) * float64(len(booster.tree[[2]int64{depth + 1, 2 * nodeIndex}].samplesId))
				right := booster.TreeShapSubset(X, sampleId, subsetIndex, depth+1, 2*nodeIndex+1) * float64(len(booster.tree[[2]int64{depth + 1, 2*nodeIndex + 1}].samplesId))
				return (left + right) / float64(len(node.samplesId))
			}
		} else {
			//此处为node.featureOwner==“host"
			if subsetIndex[int64(len(XI))+node.featureIndex] == 1 {
				ifLeft := FindNextNode(booster.ctx, booster.client, booster.id, int64(math.Exp2(float64(depth)))+nodeIndex-1, []int64{sampleId})[0]
				if ifLeft {
					return booster.TreeShapSubset(X, sampleId, subsetIndex, depth+1, 2*nodeIndex)
				} else {
					return booster.TreeShapSubset(X, sampleId, subsetIndex, depth+1, 2*nodeIndex+1)
				}
			} else {
				left := booster.TreeShapSubset(X, sampleId, subsetIndex, depth+1, 2*nodeIndex) * float64(len(booster.tree[[2]int64{depth + 1, 2 * nodeIndex}].samplesId))
				right := booster.TreeShapSubset(X, sampleId, subsetIndex, depth+1, 2*nodeIndex+1) * float64(len(booster.tree[[2]int64{depth + 1, 2*nodeIndex + 1}].samplesId))
				return (left + right) / float64(len(node.samplesId))
			}
		}
	}
}

func ConvInt2Byte(x int64) []int64 {
	var res []int64
	for {
		if x/2 == 0 {
			res = append([]int64{x}, res...)
			return res
		} else {
			res = append([]int64{x % 2}, res...)
			x = x / 2
		}
	}
}

func Byte2Int(x []int64) int64 {
	res := int64(0)
	for i := range x {
		res += x[len(x)-i-1] * int64(int(math.Exp2(float64(i))))
	}
	return res
}

func Counter1(x []int64) int64 {
	res := int64(0)
	for _, v := range x {
		res += v
	}
	return res
}

func Subset(length, index int64) [][]int64 {
	//生成长度为length， 第index个位置为0的切片，表示共有length个特征，第index特征为空
	res := make([][]int64, int64(int(math.Exp2(float64(length-1)))))
	index = length - index - 1
	index2 := int64(int(math.Exp2(float64(index))))
	i := int64(0)
	j := int64(0)
	counter := int64(0)
	for {
		if i == int64(len(res)) {
			break
		}
		if j == index2 {
			j = 0
			counter++
		}
		subset := ConvInt2Byte(i + counter*(index2))
		subsetLength := int64(len(subset))
		for k := int64(0); k < length-subsetLength; k++ {
			subset = append([]int64{0}, subset...)
		}
		res[i] = subset
		i++
		j++
	}
	return res
}

func (booster Booster) TreeShap(X [][]float64) [][]float64 {
	length := int64(len(X[0])) + booster.nHostFeatures
	SHAP := make([][]float64, len(X))
	for i := range X {
		m := map[int64]float64{}
		for j := int64(0); j < int64(math.Exp2(float64(length))); j++ {
			subsetIndex := ConvInt2Byte(j)
			lengthSubsetIndex := len(subsetIndex)
			for k := 0; int64(k) < length-int64(lengthSubsetIndex); k++ {
				subsetIndex = append([]int64{0}, subsetIndex...)
			}
			m[j] = booster.TreeShapSubset(X, int64(i), subsetIndex, 0, 0)
		}
		var Phi []float64
		for j := int64(0); j < length; j++ {
			phi := 0.0
			subsets := Subset(length, j)
			for k := range subsets {
				phi += float64(factorial(Counter1(subsets[k]))*factorial(length-Counter1(subsets[k])-1)) / float64(factorial(length)) * (m[Byte2Int(subsets[k])+int64(math.Exp2(float64(length-j-1)))] - m[Byte2Int(subsets[k])])
			}
			Phi = append(Phi, phi)
		}
		SHAP[i] = Phi
	}
	return SHAP
}

func factorial(a int64) int64 {
	if a == 0 {
		return 1
	} else if a > 0 {
		return a * factorial(a-1)
	} else {
		_ = errors.New("不符合要求")
		return 0
	}
}

type Secureboost struct {
	MaxDepth int64
	// maxDepth 包括根节点，不包含叶子节点
	NBoosters     int64
	Lambda        float64
	LearningRate  float64
	Gamma         float64
	Model         []Booster
	Client        hostGRPC.HostServiceClient
	Ctx           context.Context
	NHostFeatures int64
	SHAPPhi0      float64
}

func (secureboost *Secureboost) Fit(X [][]float64, y []int64) {
	sxgb := *new([]Booster)
	for i := int64(0); i < secureboost.NBoosters; i++ {
		if i == 0 {
			score := make([]float64, len(y))
			for j := 0; j < len(y); j++ {
				score[j] = secureboost.LearningRate * 0.0
			}
			booster := Booster{maxDepth: secureboost.MaxDepth, lambda: secureboost.Lambda, gamma: secureboost.Gamma, id: i, client: secureboost.Client, ctx: secureboost.Ctx, nHostFeatures: secureboost.NHostFeatures, learningRate: secureboost.LearningRate}
			booster.fit(X, y, score)
			sxgb = append(sxgb, booster)
		} else {
			score := make([]float64, len(y))
			for _, booster := range sxgb {
				boosterWeight := booster.predict(X)
				for j := 0; j < len(y); j++ {
					score[j] = score[j] + boosterWeight[j]
				}
			}
			for j := 0; j < len(y); j++ {
				score[j] = secureboost.LearningRate * (0.0 + score[j])
			}
			booster := Booster{maxDepth: secureboost.MaxDepth, lambda: secureboost.Lambda, gamma: secureboost.Gamma, id: i, client: secureboost.Client, ctx: secureboost.Ctx, nHostFeatures: secureboost.NHostFeatures, learningRate: secureboost.LearningRate}
			fmt.Println("New booster begin fitting")
			booster.fit(X, y, score)
			sxgb = append(sxgb, booster)
		}
	}
	secureboost.Model = sxgb
	score := make([]float64, len(X))
	for _, booster := range secureboost.Model {
		boosterWeight := booster.predict(X)
		for j := 0; j < len(X); j++ {
			score[j] = score[j] + boosterWeight[j]
		}
	}
	for j := 0; j < len(X); j++ {
		score[j] = secureboost.LearningRate * (0.0 + score[j])
	}
	SHAPPhi0 := 0.0
	for _, v := range score {
		SHAPPhi0 += v
	}
	secureboost.SHAPPhi0 = SHAPPhi0 / float64(len(score))
}

func (secureboost Secureboost) PredictProba(X [][]float64) []float64 {
	score := make([]float64, len(X))
	for _, booster := range secureboost.Model {
		boosterWeight := booster.predict(X)
		for j := 0; j < len(X); j++ {
			score[j] = score[j] + boosterWeight[j]
		}
	}
	for j := 0; j < len(X); j++ {
		score[j] = secureboost.LearningRate * (0.0 + score[j])
	}
	proba := Sigmoid(score)
	return proba
}

func (secureboost Secureboost) Predict(X [][]float64) []int {
	proba := secureboost.PredictProba(X)
	y := make([]int, len(proba))
	for i := 0; i < len(proba); i++ {
		if proba[i] >= 0.5 {
			y[i] = 1
		} else {
			y[i] = 0
		}
	}
	return y
}

func (secureboost Secureboost) TreeSHAP(X [][]float64) [][]float64 {
	SHAP := make([][][]float64, secureboost.NBoosters)
	length := int64(len(X[0])) + secureboost.NHostFeatures
	for i := range SHAP {
		SHAP[i] = secureboost.Model[i].TreeShap(X)
	}
	TreeSHAP := make([][]float64, len(X))
	for i := range TreeSHAP {
		lines := make([]float64, length)
		for j := int64(0); j < length; j++ {
			tmpSHAP := 0.0
			for k := int64(0); k < secureboost.NBoosters; k++ {
				tmpSHAP += SHAP[k][i][j]
			}
			lines[j] = tmpSHAP
		}
		TreeSHAP[i] = append(lines, secureboost.SHAPPhi0)
	}
	return TreeSHAP
}

func Sigmoid(x []float64) []float64 {
	res := make([]float64, len(x))
	for i := 0; i < len(x); i++ {
		res[i] = 1 / (1 + math.Exp(-x[i]))
	}
	return res
}
