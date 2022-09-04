import functools

import torch


def content_loss(content_features, output_features, L2):

    dimension = content_features.shape[1:]
    denominator = functools.reduce((lambda x, y: x * y), dimension)
    return (
        torch.stack(
            [
                L2(content_feature, output_feature)
                for (content_feature, output_feature) in zip(
                    content_features, output_features
                )
            ]
        ).sum()
        / denominator
    )


def gram_matrix(x_feature):

    a, b, c, d = x_feature.size()
    features = x_feature.view(a * b, c * d)
    G = torch.mm(
        features.to(torch.float64), features.t().to(torch.float64)
    )  # may easily overflow without float64 :(
    return G.div(a * b * c * d)


def style_loss(style_features, output_features, L2, device):
    num_layers = style_features.shape[1]
    style_matrices = gram_matrix(style_features).to(torch.float16 if device == "cuda" else torch.float32)
    output_matrices = gram_matrix(output_features).to(torch.float16 if device == "cuda" else torch.float32)
    return (
        torch.stack(
            [
                L2(style_matrix, output_matrix)
                for (style_matrix, output_matrix) in zip(
                    style_matrices, output_matrices
                )
            ]
        ).sum()
        / num_layers
    )


def regularizer(output_imgs, L2):
    return torch.stack(
        [
            L2(img[:, :-1, :], img[:, 1:, :])
            + L2(img[:, :, :-1], img[:, :, 1:])
            for img in output_imgs
        ]
    ).sum()


# substitute for replacing optical flow in temporal loss
# main idea is to suppress noise in different frames
# https://arxiv.org/abs/1604.08610
def noise_loss(output_imgs, noisy_input_output_imgs, L2):
    return torch.stack(
        [
            L2(output_img, noisy_input_output_img)
            for (output_img, noisy_input_output_img) in zip(
                output_imgs, noisy_input_output_imgs
            )
        ]
    ).sum()
