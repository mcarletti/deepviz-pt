import numpy as np
import torch


def softmax(x):
    exp = np.exp(x)
    return exp / np.sum(exp)


def predictions_to_class_info(pred):
    pred = softmax(pred)
    class_id = np.argmax(pred)
    class_prob = pred[class_id]
    class_labels = np.loadtxt(open('data/ilsvrc_2012_labels.txt'), dtype=object, delimiter='\n')
    return class_id, class_labels[class_id], pred[class_id]


def compute_saliency_map(model, tensor_image, k_size, ref_class_id, ref_class_prob):
    assert k_size >= 3 and k_size % 2 == 1

    saliency_values = []
    ch, rows, cols = tensor_image.shape
    h_size = int(k_size / 2)

    for u in range(h_size, rows, h_size):
        for v in range(h_size, cols, h_size):
            masked_image = tensor_image.clone()
            mask_color = torch.mean(masked_image[:, u-h_size:u+h_size, v-h_size:v+h_size])
            masked_image[:, u-h_size:u+h_size, v-h_size:v+h_size] = mask_color
            img = masked_image.unsqueeze(0)
            pred = model(torch.autograd.Variable(img)) # 1000 scores on gpu
            pred = pred.data.cpu().numpy().squeeze() # ... on cpu
            pred_class_id = np.argmax(pred)
            pred = 0.0 if ref_class_id != pred_class_id else softmax(pred)[pred_class_id]
            pred_err = ref_class_prob - pred
            if np.abs(pred_err) < 0.05:
                pred_err = 0.0
            saliency_values.append(pred_err)
    
    size = int(np.sqrt(len(saliency_values)))
    saliency_map = np.array(saliency_values, dtype=np.float32)
    saliency_map = saliency_map.reshape((size, size))

    return saliency_map
