import os
import logging
import time
import cv2
import numpy as np
import torch
from moviepy.editor import VideoFileClip, concatenate_videoclips

from helpers import init_helper, bbox_helper, video_helper, vsumm_helper
from modules.model_zoo import get_model
from generate_summary import generate_summary


CHECKPOINT_PATHS = {
    'pglsum': '../models/pglsum.pt',
    'casum': '../models/casum.pt',
    'dsnet_ab': '../models/dsnet_ab.pt',
    'dsnet_af': '../models/dsnet_af.pt',
}

logger = logging.getLogger()


def get_boundaries(list_of_frames_binary):
    result = []
    previous_digit = 0
    digit = 0
    position = 0

    for position, digit in enumerate(list_of_frames_binary):
        if (digit == 1) and (previous_digit == 0):
            result.append([position])
            previous_digit = 1

        elif (digit == 0) and (previous_digit == 1):
            result[-1].append(position - 1)
            previous_digit = 0

        else:
            continue

    if digit == 1:
        result[-1].append(position)

    return result


def save_output_video(source, boundaries, save_path, prediction_summary, write_with_moviepy=True):
    # write_with_moviepy if True write video summary with audio
    if write_with_moviepy:
        logger.info('Moviepy write summary')
        clip = VideoFileClip(source)

        # clip list
        clips = [clip.subclip(scene[0], scene[1]) for scene in boundaries]

        # concatenating both the clips
        final_with_audio = concatenate_videoclips(clips)
        final = final_with_audio.without_audio()
        final.write_videofile(save_path)

    else:
        logger.info('cv2 write summary')
        logger.info(f'prediction_summary len: {len(prediction_summary)}')

        # load original video
        cap = cv2.VideoCapture(source)
        width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        fps = cap.get(cv2.CAP_PROP_FPS)

        # create summary video writer
        fourcc = cv2.VideoWriter_fourcc(*'mp4v')
        out = cv2.VideoWriter(save_path, fourcc, fps, (width, height))

        frame_idx = 0
        while True:
            ret, frame = cap.read()
            if not ret:
                break

            if frame_idx == len(prediction_summary):
                break

            if prediction_summary[frame_idx]:
                out.write(frame)

            frame_idx += 1

        out.release()
        cap.release()

    logger.info(f'Saved to {save_path}')


def load_model(args):
    model = get_model(args.model, **vars(args))
    model = model.eval().to(args.device)
    checkpoint_path = CHECKPOINT_PATHS[args.model]
    state_dict = torch.load(checkpoint_path,
                            map_location=lambda storage, loc: storage)
    model.load_state_dict(state_dict)
    return model


def main():
    args = init_helper.get_arguments()

    init_helper.init_logger(args.model_dir, args.log_file)
    init_helper.set_random_seed(args.seed)

    logger.info(vars(args))

    start_time_main = time.time()
    last_step_time = time.time()
    args = init_helper.get_arguments()

    # load model
    model = load_model(args)
    logger.info(f'Loading model {args.model} time: {time.time() - last_step_time}')

    last_step_time = time.time()

    # creating lists of filepaths to source videos and to filepaths to save summary
    if os.path.isfile(args.source):
        videos = [args.source]
        save_path_list = [args.save_path]

        save_dir = args.save_path.rsplit(sep='/', maxsplit=1)[0]
        if not os.path.exists(save_dir):
            os.makedirs(save_dir)

    else:
        videos = [video for video in os.listdir(args.source) if video.endswith('.mp4')]
        save_path_list = [os.path.join(args.save_path,
                                       args.model.lower() + '_' + video_name) for video_name in videos]
        videos = [os.path.join(args.source, video) for video in videos]

        if not os.path.exists(args.save_path):
            os.makedirs(args.save_path)

    logger.info(f"videos: {' | '.join(videos)}")
    logger.info(f"save_path_list: {' | '.join(save_path_list)}")

    for video_source, save_path in zip(videos, save_path_list):
        # load video and extract features
        video_start_time = time.time()
        logger.info(f'\nProcessing video: {video_source}')
        video_proc = video_helper.VideoPreprocessor(args.sample_rate)
        n_frames, frame_features, cps, nfps, picks, fps = video_proc.run(video_source,
                                                                         args.max_shot_length * 30)
        frame_features_len = len(frame_features)

        logger.info(cps)

        logger.info(f'Preprocessing source video time: {time.time() - last_step_time}')
        last_step_time = time.time()

        # predicting summary
        with torch.no_grad():
            if args.model.lower() in ['dsnet_ab', 'dsnet_af']:
                seq_torch = torch.from_numpy(frame_features).unsqueeze(0).to(args.device)
                pred_cls, pred_bboxes = model.predict(seq_torch)
                pred_bboxes = np.clip(pred_bboxes, 0, frame_features_len).round().astype(np.int32)

                pred_cls, pred_bboxes = bbox_helper.nms(pred_cls, pred_bboxes, args.nms_thresh)
                scores = vsumm_helper.bbox2scores(
                    frame_features_len, pred_cls, pred_bboxes)

                pred_summ = generate_summary([cps], [scores], [n_frames], [picks],
                                             args.final_frame_length * 30,
                                             args.min_penalty_shot_length)[0]

            elif args.model.lower() in ['casum', 'pglsum']:
                frame_features = torch.Tensor(frame_features).view(-1, 1024)
                frame_features = frame_features.to(model.linear_1.weight.device)

                scores, _ = model(frame_features)  # [1, seq_len]
                scores = scores.squeeze(0).cpu().numpy().tolist()
                pred_summ = generate_summary([cps], [scores], [n_frames], [picks],
                                             args.final_frame_length * 30,
                                             args.min_penalty_shot_length * 30)[0]

            else:
                raise ValueError('Invalid model type', args.model)

        logger.info(f"len pred_summ {len(pred_summ)}, selected: {sum(pred_summ)}")
        boundaries = np.array(get_boundaries(pred_summ)) / fps
        logger.info(boundaries)

        logger.info(f"N boundaries: {len(boundaries)}")
        logger.info(f"max boundary len: {max([end - begin for (begin, end) in boundaries])}")
        logger.info(f"min boundary len: {min([end - begin for (begin, end) in boundaries])}")

        logger.info(f'Predicting summary: {time.time() - last_step_time}')
        last_step_time = time.time()

        save_output_video(video_source, boundaries, save_path, pred_summ)

        logger.info(f'Writing summary video: {time.time() - last_step_time}')

        logger.info(f'Total video {video_source} time: {time.time() - video_start_time}\n\n')

    logger.info(f'Total time: {time.time() - start_time_main}')


if __name__ == '__main__':
    main()
