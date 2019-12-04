import React from 'react';
import Card from '@material-ui/core/Card';
import CardActions from '@material-ui/core/CardActions';
import CardContent from '@material-ui/core/CardContent';
import CardHeader from '@material-ui/core/CardHeader';
import CssBaseline from '@material-ui/core/CssBaseline';
import Grid from '@material-ui/core/Grid';
import StarIcon from '@material-ui/icons/StarBorder';
import Typography from '@material-ui/core/Typography';
import { makeStyles } from '@material-ui/core/styles';
import Container from '@material-ui/core/Container';
import './App.css';


const useStyles = makeStyles(theme => ({
  '@global': {
    body: {
      backgroundColor: theme.palette.common.white,
    },
    ul: {
      margin: 0,
      padding: 0,
    },
    li: {
      listStyle: 'none',
    },
  },
  appBar: {
    borderBottom: `1px solid ${theme.palette.divider}`,
  },
  toolbar: {
    flexWrap: 'wrap',
  },
  toolbarTitle: {
    flexGrow: 1,
  },
  link: {
    margin: theme.spacing(1, 1.5),
  },
  heroContent: {
    padding: theme.spacing(8, 0, 6),
  },
  cardHeader: {
    backgroundColor: theme.palette.grey[200],
  },
  cardPricing: {
    display: 'flex',
    justifyContent: 'center',
    alignItems: 'baseline',
    marginBottom: theme.spacing(2),
  },
  footer: {
    borderTop: `1px solid ${theme.palette.divider}`,
    marginTop: theme.spacing(8),
    paddingTop: theme.spacing(3),
    paddingBottom: theme.spacing(3),
    [theme.breakpoints.up('sm')]: {
      paddingTop: theme.spacing(6),
      paddingBottom: theme.spacing(6),
    },
  },
}));

const tiers = [
  {
    title: 'BPM',
    description: ['Select Beats Per Minute'],
    buttonText: 'Contact us',
  },
  {
    title: 'Genre',
    description: ['Select Genre'],
    buttonText: 'Contact us',              
  },
  {
    title: 'Sub-Genre',
    description: ['Select Sub-Genre'],
    buttonText: 'Contact us',
  },
];


export default function Pricing() {
  const classes = useStyles();

  return (
    <React.Fragment>
      <CssBaseline />

    <div className= "TextBPM">
       <input type="range" min="1" max="100" value="50" class="slider" id="myRange"/>
    </div>

    <div className= "TextGenre">
      <select>
        <option value="Rap">Rap</option>
        <option value="Hip-Hop">Hip-Hop</option>
        <option value="Electronic">Electronic</option>
        <option value="R and B">R&B</option>
      </select>
      </div>

    <div className= "TextSubGenre">
            <select>
        <option value="Rap">Rap</option>
        <option value="Hip-Hop">Hip-Hop</option>
        <option value="Electronic">Electronic</option>
        <option value="R and B">R&B</option>
      </select>
    </div>

      <div className = "GenerateButton">
      <input type="button" value="Generate Song!" onclick="msg()"/>
      </div>


      {/* Hero unit */}
      <Container maxWidth="sm" component="main" className={classes.heroContent}>
        <Typography component="h1" variant="h2" align="center" color="textPrimary" gutterBottom>
          yung gan
        </Typography>
        <Typography variant="h5" align="center" color="textSecondary" component="p">
         A Generative Adversarial Network Music Producer. It will use the data you enter to 
         create a beat of your choice.
        </Typography>
      </Container>
      {/* End hero unit */}


      <Container maxWidth="md" component="main">
        <Grid container spacing={5} alignItems="flex-end">
          {tiers.map(tier => (
            // Enterprise card is full width at sm breakpoint
            <Grid item key={tier.title} xs={12} sm={tier.title === 'Enterprise' ? 12 : 6} md={4}>
    
              <Card>


                <CardHeader
                  title={tier.title}
                  subheader={tier.subheader}
                  titleTypographyProps={{ align: 'center' }}
                  subheaderTypographyProps={{ align: 'center' }}
                  action={tier.title === 'Pro' ? <StarIcon /> : null}
                  className={classes.cardHeader}
                />


                <CardContent>
                  <div className={classes.cardPricing}>
                  
                  </div>
                  <ul>
                    {tier.description.map(line => (
                      <Typography component="li" variant="subtitle1" align="center" key={line}>
                        {line}
                      </Typography>
                    ))}
                  </ul>
                  
                </CardContent>


                <CardActions>
                

                </CardActions>
              </Card>
            </Grid>
          ))}
        </Grid>
      </Container>
    </React.Fragment>
  );
}
