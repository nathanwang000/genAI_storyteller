import click
from simple_youtube_api.Channel import Channel
from simple_youtube_api.LocalVideo import LocalVideo

def parse_credential(credential:str)->str:
    if len(credential.split('.')) < 2 or credential.split('.')[-1] != 'storage':
        return credential + '.storage'
    return credential

@click.command()
@click.option('--fn', type=str, prompt='video file name',
              help='video file name')
@click.option('--credential', type=str,
              prompt='channel credential file name',
              help='channel credential file name')
@click.option('--title', type=str,
              prompt='Title of the video',
              help='Title of the video')
@click.option('--description', type=str,
              prompt='Description of the video',
              help='Description of the video')
@click.option('--privacy', type=click.Choice(['public', 'private', 'unlisted']), default='public')
def main(fn, credential, title, description, privacy):
    # loggin into the channel
    channel = Channel()
    channel.login("client_secrets.json",
                  parse_credential(credential))
    #"credentials.storage")
    
    # setting up the video that is going to be uploaded
    video = LocalVideo(file_path=fn)
    
    # setting snippet
    video.set_title(title)
    video.set_description(description)
    # video.set_tags(["this", "tag"])
    # video.set_category("gaming")
    # video.set_default_language("en-US")
    
    # setting status
    video.set_embeddable(True)
    # video.set_license("creativeCommon")
    video.set_privacy_status(privacy)
    video.set_public_stats_viewable(True)
    
    # setting thumbnail
    # video.set_thumbnail_path('test_thumb.png')
    
    # uploading video and printing the results
    video = channel.upload_video(video)
    # channel.add_video_to_playlist(playlist_id, video):
    print(video.id)
    print(video)
    
    # liking video
    video.like()

if __name__ == '__main__':
    main()
